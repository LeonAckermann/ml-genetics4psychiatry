"""Unified training entry point for sklearn and DNN models."""
from __future__ import annotations

from typing import Callable

import numpy as np


def train(
    model_or_cfg,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    return_model: bool = False,
    history: list | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> np.ndarray:
    """Return test-set predictions (and, if return_model=True, the fitted model too).

    Dispatches based on model_or_cfg type:
      - dict with 'class' key and class is MDN → MDN path (returns expected value)
      - dict with 'class' key                  → PyTorch DNN path
      - anything else                           → sklearn fit / predict path

    return_model=True changes the return type from `preds` to `(preds, fitted_model)`.
    Not supported for MDN (out of scope — callers must not pass return_model=True
    when model_or_cfg builds an MDN).

    ``history``: pass a list to have one record per epoch (or, for XGBoost, per
    boosting round) appended to it -- train/test loss + score. Passed as a
    mutable out-parameter rather than a return value so the return arity stays
    fixed for every caller. Populated for DNN, MDN, and XGBoost (regardless of
    whether XGBoost's early stopping is configured); every other sklearn model
    has no per-step curve and leaves it empty.

    ``on_epoch``: optional callback invoked with that same record the moment
    it's produced -- i.e. live, during training, not batched at the end (see
    src/log.py). Ignored for models with no per-step curve.
    """
    if isinstance(model_or_cfg, dict) and "class" in model_or_cfg:
        from model import MDN
        if model_or_cfg["class"] is MDN:
            preds, _pi, _mu, _init_mu, _init_sigma = train_mdn(model_or_cfg, X_train, y_train, X_val, y_val, X_test, cfg, y_test=y_test, history=history, on_epoch=on_epoch)
            return preds
        preds, fitted = _train_dnn(model_or_cfg, X_train, y_train, X_val, y_val, X_test, cfg, y_test=y_test, history=history, on_epoch=on_epoch)
        return (preds, fitted) if return_model else preds
    preds, fitted = _train_sklearn(
        model_or_cfg, X_train, y_train, X_val, y_val, X_test,
        y_test=y_test,
        task_type=cfg.get("model", {}).get("type", "regression"),
        on_epoch=on_epoch,
        history=history,
    )
    return (preds, fitted) if return_model else preds


# ---------------------------------------------------------------------------
# Per-epoch evaluation (training curves)
# ---------------------------------------------------------------------------

def _mdn_epoch_scores(model, loader, nll_fn, device) -> tuple[float, float]:
    """MDN counterpart of ``_epoch_scores``: mixture NLL, and R² of E[y]."""
    import torch

    model.eval()
    total_loss, n = 0.0, 0
    preds, ys = [], []
    with torch.no_grad():
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            pi, mu, sigma = model(bX)
            total_loss += nll_fn(pi, mu, sigma, by).item() * bX.size(0)
            n += bX.size(0)
            preds.append((pi * mu).sum(dim=1).detach().cpu())
            ys.append(by.detach().cpu())

    loss = total_loss / max(n, 1)
    if not preds:
        return loss, float("nan")
    pred = torch.cat(preds).numpy().ravel()
    true = torch.cat(ys).numpy().ravel()
    ss_tot = float(((true - true.mean()) ** 2).sum())
    ss_res = float(((true - pred) ** 2).sum())
    return loss, (float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))


def _epoch_scores(model, loader, criterion, task_type, device) -> tuple[float, float]:
    """Loss and task score over one whole split, in eval mode.

    Evaluated with dropout off so the train curve is directly comparable to the
    test curve — a running average of the mini-batch losses would be measured
    mid-update and under dropout instead. Score is accuracy for classification,
    R² for regression.
    """
    import torch

    model.eval()
    total_loss, n = 0.0, 0
    outs, ys = [], []
    with torch.no_grad():
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            out = model(bX)
            total_loss += criterion(out, by).item() * bX.size(0)
            n += bX.size(0)
            outs.append(out.detach().cpu())
            ys.append(by.detach().cpu())

    loss = total_loss / max(n, 1)
    pred = torch.cat(outs).numpy().ravel() if outs else np.empty(0)
    true = torch.cat(ys).numpy().ravel() if ys else np.empty(0)
    if pred.size == 0:
        return loss, float("nan")

    if task_type in ("classification", "binary_classification"):
        # model emits logits (BCEWithLogitsLoss); threshold at p = 0.5
        score = float(((pred > 0.0).astype(np.float32) == true).mean())
    else:
        ss_tot = float(((true - true.mean()) ** 2).sum())
        ss_res = float(((true - pred) ** 2).sum())
        score = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return loss, score


def train_mdn(
    model_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: dict,
    y_test: np.ndarray | None = None,
    history: list | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train an MDN and return (expected_value_preds, pi, mu) on X_test.

    expected_value_preds  — weighted mean (sum pi*mu), shape (N,)
    pi                    — mixing weights, shape (N, n_components)
    mu                    — component means, shape (N, n_components)

    Samples with y < 0 have both X and y sign-flipped before training so the
    MDN sees a unimodal (always-positive) distribution.  The same flip is
    applied to the test set (requires y_test to be provided).
    """
    import torch
    import torch.distributions as D
    from torch import optim
    from torch.utils.data import DataLoader
    from dataloader import GWASDataset

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    verbose = cfg.get("verbose", True)

    # ── Sign-flip: invert samples where y < 0 ────────────────────────────────
    def _flip(X: np.ndarray, y: np.ndarray):
        X = np.array(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        neg = y < 0
        X[neg] *= -1
        y[neg] *= -1
        return X, y

    X_train_f, y_train_f = _flip(X_train, y_train)
    X_val_f, y_val_f = _flip(X_val, y_val)
    if y_test is not None:
        X_test_f, _ = _flip(X_test, y_test)
    else:
        X_test_f = np.asarray(X_test, dtype=np.float32)

    # Data-driven initialisation: split at Z=2 (non-significant vs significant),
    # matching the notebook convention.  For K>2 the significant group is further
    # split into K-1 equal quantile bins.
    n_components = model_cfg["number_of_components"]
    non_sig = y_train_f[y_train_f < 2]
    sig     = y_train_f[y_train_f >= 2]
    if len(non_sig) == 0:
        non_sig = y_train_f
    if len(sig) == 0:
        sig = y_train_f
    if n_components == 2:
        groups = [non_sig, sig]
    else:
        sig_bins = np.array_split(np.sort(sig), n_components - 1)
        groups = [non_sig] + sig_bins
    init_mu = [float(np.median(g)) for g in groups]
    # inverse-softplus: b such that softplus(b) ≈ s  →  b = log(exp(s) - 1)
    init_sigma_bias = [float(np.log(np.expm1(max(g.std(), 1e-2)))) for g in groups]
    # Convert bias → actual sigma (same transform as MDNOutputLayer.forward)
    init_sigma = [float(np.log1p(np.exp(s)) + 1e-6) for s in init_sigma_bias]

    model = model_cfg["class"](
        input_dim=X_train_f.shape[1],
        hidden_dims=model_cfg["hidden_dims"],
        output_dim=1,
        dropout=model_cfg.get("dropout"),
        number_of_components=n_components,
        random_state=model_cfg.get("seed", 42),
        mu=init_mu,
        sigma=init_sigma_bias,
    ).to(device)

    batch_size = model_cfg["batch_size"]
    train_loader = DataLoader(
        GWASDataset(X_train_f, y_train_f), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        GWASDataset(X_val_f, y_val_f), batch_size=batch_size, shuffle=False, drop_last=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg.get("weight_decay", 0.0),
    )

    def _mdn_nll(pi, mu, sigma, target):
        target = target.view(-1, 1)
        log_prob = D.Normal(loc=mu, scale=sigma).log_prob(target)
        log_pi = torch.log(pi + 1e-8)
        return -torch.mean(torch.logsumexp(log_prob + log_pi, dim=1))

    # Curves are recorded on the sign-flipped splits the MDN actually sees.
    record = history is not None and y_test is not None
    if record:
        _, y_test_f = _flip(X_test, y_test)
        curve_train_loader = DataLoader(
            GWASDataset(X_train_f, y_train_f), batch_size=batch_size, shuffle=False, drop_last=False
        )
        curve_test_loader = DataLoader(
            GWASDataset(X_test_f, y_test_f), batch_size=batch_size, shuffle=False, drop_last=False
        )

    patience = int(model_cfg.get("patience", 20))
    best_val_loss = float("inf")
    best_weights: dict | None = None
    no_improve = 0

    for epoch in range(model_cfg["epochs"]):
        model.train()
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            pi, mu, sigma = model(bX)
            loss = _mdn_nll(pi, mu, sigma, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vX, vy in val_loader:
                vX, vy = vX.to(device), vy.to(device)
                pi_v, mu_v, sigma_v = model(vX)
                val_loss += _mdn_nll(pi_v, mu_v, sigma_v, vy).item() * vX.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        if record:
            tr_loss, tr_r2 = _mdn_epoch_scores(model, curve_train_loader, _mdn_nll, device)
            te_loss, te_r2 = _mdn_epoch_scores(model, curve_test_loader, _mdn_nll, device)
            record_dict = {
                "epoch": epoch + 1,
                "train_loss": tr_loss,          # mixture NLL, not MSE
                "test_loss": te_loss,
                "train_r2": tr_r2,              # R² of the mixture expected value
                "test_r2": te_r2,
                "val_loss": val_loss,
                "score_name": "r2",
            }
            history.append(record_dict)
            if on_epoch is not None:
                on_epoch(record_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    MDN early stopping at epoch {epoch + 1} (val_nll={best_val_loss:.4f})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    X_test_t = torch.tensor(X_test_f).to(device)
    with torch.no_grad():
        pi_t, mu_t, _sigma_t = model(X_test_t)

    pi_np = pi_t.cpu().numpy()   # (N, K)
    mu_np = mu_t.cpu().numpy()   # (N, K)
    expected_value = (pi_t * mu_t).sum(dim=1).cpu().numpy()  # (N,)

    return expected_value, pi_np, mu_np, init_mu, init_sigma


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _xgb_native(model):
    """Return the underlying xgboost estimator, or None if this isn't one.

    build_model returns either a bare XGBRegressor (early-stopping path) or a
    model/ wrapper holding one in `.model`.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return None
    native = getattr(model, "model", model)
    return native if isinstance(native, (xgb.XGBRegressor, xgb.XGBClassifier)) else None


def _xgb_curve_records(native, y_train, y_val, y_test, task_type) -> list[dict]:
    """Turn xgboost's evals_result() into the same per-step records the DNN emits.

    Boosting rounds stand in for epochs. The scores come free from the eval
    metrics rather than from re-predicting at every round:
      * regression — rmse over a split is exactly sqrt(MSE), so
        R² = 1 - rmse²/var(y) for that split, no extra passes.
      * classification — xgboost's 'error' metric is 1 - accuracy.
    """
    res = native.evals_result()
    if not res:
        return []
    return _xgb_records_from_evals(res, y_train, y_val, y_test, task_type)


def _xgb_records_from_evals(res, y_train, y_val, y_test, task_type) -> list[dict]:
    """Shared core of _xgb_curve_records -- also called, growing by one round
    each time, from _XgbLiveCallback for live (in-training) logging. `res` is
    xgboost's evals_result()-shaped dict; a callback's `evals_log` argument has
    the identical shape and can be passed here directly."""
    # Eval-set order is fixed by _train_sklearn below: val first, so that
    # validation_0 stays the set the early-stopping callback watches.
    named = {"val": res.get("validation_0", {}),
             "train": res.get("validation_1", {}),
             "test": res.get("validation_2", {})}
    is_cls = task_type in ("classification", "binary_classification")
    loss_key = "logloss" if is_cls else "rmse"
    if loss_key not in named["train"]:
        loss_key = next(iter(named["train"]), None)
        if loss_key is None:
            return []

    score_name = "accuracy" if is_cls else "r2"
    variances = {"train": float(np.var(np.asarray(y_train, dtype=float))),
                 "test": float(np.var(np.asarray(y_test, dtype=float))),
                 "val": float(np.var(np.asarray(y_val, dtype=float)))}

    def score_at(split: str, i: int) -> float:
        m = named[split]
        if is_cls:
            err = m.get("error")
            return float(1.0 - err[i]) if err is not None and i < len(err) else float("nan")
        rmse = m.get("rmse")
        var = variances[split]
        if rmse is None or i >= len(rmse) or var <= 0:
            return float("nan")
        return float(1.0 - (rmse[i] ** 2) / var)

    n_rounds = len(named["train"].get(loss_key, []))
    records = []
    for i in range(n_rounds):
        rec = {
            "epoch": i + 1,
            "train_loss": float(named["train"][loss_key][i]),
            "test_loss": float(named["test"][loss_key][i]) if named["test"].get(loss_key) else float("nan"),
            f"train_{score_name}": score_at("train", i),
            f"test_{score_name}": score_at("test", i),
            "score_name": score_name,
            "step_name": "boosting round",
        }
        val_series = named["val"].get(loss_key)
        if val_series is not None and i < len(val_series):
            rec["val_loss"] = float(val_series[i])
        records.append(rec)
    return records


def _make_xgb_live_callback(on_epoch, y_train, y_val, y_test, task_type):
    """Build an xgboost TrainingCallback that fires ``on_epoch`` once per
    boosting round, live during ``.fit()`` -- xgboost's C++ training loop only
    returns to Python at each round boundary via this hook, so this is the
    only way to log boosting-round metrics as they happen rather than after
    the whole fit. Reuses _xgb_records_from_evals on xgboost's own growing
    `evals_log`, so the live and post-hoc (_xgb_curve_records) records are
    computed identically. Class is defined here (not at module level) so the
    xgboost import stays lazy, matching the rest of this module."""
    import xgboost as xgb

    class _LiveCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            records = _xgb_records_from_evals(evals_log, y_train, y_val, y_test, task_type)
            if records:
                on_epoch(records[-1])
            return False  # False = don't stop training

    return _LiveCallback()


def _train_sklearn(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | None = None,
    task_type: str = "regression",
    history: list | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, object]:
    """Fit an sklearn-compatible model and return (predictions on X_test, fitted model).

    For XGBoost, X_val is passed as the eval_set so early stopping can fire, and
    — when ``history`` is requested — the outer train and test splits are added
    as further eval sets to record per-round curves. Validation stays **first**
    (validation_0) because the early-stopping callback in src/hpo.py pins
    ``data_name="validation_0"``; putting the test split there instead would
    make early stopping select on the test set. Every other model family fits
    plainly and records nothing (no iteration axis to record).

    ``on_epoch``, if given, is additionally invoked once per boosting round,
    live during ``.fit()`` (see _make_xgb_live_callback) -- separate from
    ``history``, which is only fully populated once ``.fit()`` returns.
    """
    native = _xgb_native(model)
    want_curves = history is not None and native is not None and y_test is not None

    if want_curves:
        is_cls = task_type in ("classification", "binary_classification")
        # 'error' is what accuracy is derived from; 'rmse' is what the existing
        # early-stopping callback watches, so it must stay present.
        native.set_params(eval_metric=["logloss", "error"] if is_cls else ["rmse"])
        if on_epoch is not None:
            live_cb = _make_xgb_live_callback(on_epoch, y_train, y_val, y_test, task_type)
            native.set_params(callbacks=[*(native.callbacks or []), live_cb])
        native.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val), (X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        history.extend(_xgb_curve_records(native, y_train, y_val, y_test, task_type))
    elif native is not None and getattr(native, "callbacks", None):
        native.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    preds = np.asarray(model.predict(X_test), dtype=float).ravel()
    return preds, model


def _train_dnn(
    model_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: dict,
    y_test: np.ndarray | None = None,
    history: list | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, object]:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from dataloader import GWASDataset

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    task_type = cfg.get("model", {}).get("type", "regression")
    verbose = cfg.get("verbose", True)

    model = model_cfg["class"](
        input_dim=X_train.shape[1],
        hidden_dims=model_cfg["hidden_dims"],
        output_dim=model_cfg["output_dim"],
        dropout=model_cfg["dropout"],
        random_state=model_cfg.get("seed", 42),
    ).to(device)

    batch_size = model_cfg["batch_size"]
    train_loader = DataLoader(
        GWASDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        GWASDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False
    )

    criterion = (
        nn.BCEWithLogitsLoss()
        if task_type in ("classification", "binary_classification")
        else nn.MSELoss()
    )
    optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg.get("weight_decay", 0.0))

    # Curve loaders: the outer-fold train set unshuffled/undropped (so every row
    # counts) and the outer-fold TEST set. The test curve is diagnostic only —
    # early stopping and best-weight selection below still use val_loss, so
    # nothing about the test split feeds back into training.
    record = history is not None and y_test is not None
    if record:
        curve_train_loader = DataLoader(
            GWASDataset(X_train, y_train), batch_size=batch_size, shuffle=False, drop_last=False
        )
        curve_test_loader = DataLoader(
            GWASDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False
        )
        score_name = ("accuracy" if task_type in ("classification", "binary_classification")
                      else "r2")

    patience = int(model_cfg.get("patience", 20))
    best_val_loss = float("inf")
    best_weights: dict | None = None
    no_improve = 0

    for epoch in range(model_cfg["epochs"]):
        model.train()
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vX, vy in val_loader:
                vX, vy = vX.to(device), vy.to(device)
                val_loss += criterion(model(vX), vy).item() * vX.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        if record:
            # Before the early-stopping break, so the final epoch is recorded.
            tr_loss, tr_score = _epoch_scores(model, curve_train_loader, criterion, task_type, device)
            te_loss, te_score = _epoch_scores(model, curve_test_loader, criterion, task_type, device)
            record_dict = {
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "test_loss": te_loss,
                f"train_{score_name}": tr_score,
                f"test_{score_name}": te_score,
                "val_loss": val_loss,          # the quantity early stopping acts on
                "score_name": score_name,
            }
            history.append(record_dict)
            if on_epoch is not None:
                on_epoch(record_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch + 1} (val_loss={best_val_loss:.4f})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    X_test_t = torch.tensor(np.asarray(X_test, dtype=np.float32)).to(device)
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy().flatten()
    return preds, model
