"""Unified training entry point for sklearn and DNN models."""
from __future__ import annotations

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
) -> np.ndarray:
    """Return test-set predictions.

    Dispatches based on model_or_cfg type:
      - dict with 'class' key and class is MDN → MDN path (returns expected value)
      - dict with 'class' key                  → PyTorch DNN path
      - anything else                           → sklearn fit / predict path
    """
    if isinstance(model_or_cfg, dict) and "class" in model_or_cfg:
        from model import MDN
        if model_or_cfg["class"] is MDN:
            preds, _pi, _mu, _init_mu, _init_sigma = train_mdn(model_or_cfg, X_train, y_train, X_val, y_val, X_test, cfg, y_test=y_test)
            return preds
        return _train_dnn(model_or_cfg, X_train, y_train, X_val, y_val, X_test, cfg)
    return _train_sklearn(model_or_cfg, X_train, y_train, X_val, y_val, X_test)


def train_mdn(
    model_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: dict,
    y_test: np.ndarray | None = None,
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

def _train_sklearn(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit an sklearn-compatible model and return predictions on X_test.

    For XGBRegressor instances that carry early-stopping callbacks, X_val is
    passed as the eval_set so early stopping can fire.  All other models ignore
    X_val and y_val.
    """
    try:
        import xgboost as xgb
        _XGBReg = xgb.XGBRegressor
    except ImportError:
        _XGBReg = type(None)

    if isinstance(model, _XGBReg) and getattr(model, "callbacks", None):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return np.asarray(model.predict(X_test), dtype=float).ravel()


def _train_dnn(
    model_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: dict,
) -> np.ndarray:
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
    return preds
