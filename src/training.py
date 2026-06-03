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
      - dict with 'class' key → PyTorch DNN path
      - anything else         → sklearn fit / predict path
    """
    if isinstance(model_or_cfg, dict) and "class" in model_or_cfg:
        return _train_dnn(model_or_cfg, X_train, y_train, X_val, y_val, X_test, cfg)
    return _train_sklearn(model_or_cfg, X_train, y_train, X_val, y_val, X_test)


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
    optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"])

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
