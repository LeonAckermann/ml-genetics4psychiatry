"""Per-fold SHAP/shapiq explanations for the outer-fold test set.

Called from src.cv.nested_cv after a fold's final model is trained (never
during inner-fold HPO). Builds one shap.Explanation per fold — using a
model-family-appropriate explainer — then renders every requested plot type
from that single Explanation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Models explained via shap.LinearExplainer on the unwrapped sklearn estimator.
LINEAR_MODELS: frozenset[str] = frozenset({
    "linear_regression", "lasso_regression", "ridge_regression", "elastic_regression",
    "logistic_regression", "ridge_logistic_regression", "lasso_logistic_regression",
    "elastic_logistic_regression",
})

# Models explained via shap.TreeExplainer.
TREE_MODELS: frozenset[str] = frozenset({"xgboost"})

# Torch nn.Module models. Explained via a model-agnostic Permutation
# explainer, not shap.DeepExplainer/GradientExplainer -- both raise internal
# tensor-shape errors on ResidualDNN's shortcut-connection addition (verified
# against the real ResidualBlock architecture, not just an additivity-check
# tolerance issue).
DEEP_MODELS: frozenset[str] = frozenset({"dnn", "residual_dnn"})


def _native_estimator(fitted_model):
    """Unwrap a model/ wrapper class down to the library object shap/shapiq expect.

    Handles both patterns used in model/: a `.model` attribute holding the real
    estimator (possibly a sklearn Pipeline, e.g. the logistic-regression family),
    and raw sklearn objects returned directly by build_model (linear/ridge/lasso).
    """
    from sklearn.pipeline import Pipeline

    native = getattr(fitted_model, "model", fitted_model)
    if isinstance(native, Pipeline):
        native = native.steps[-1][1]
    return native


def _subsample(X: np.ndarray, size: int, seed: int = 42) -> np.ndarray:
    if len(X) <= size:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=size, replace=False)
    return X[idx]


def _linear_explanation(fitted_model, X_background, X_test, feature_names):
    import shap

    native = _native_estimator(fitted_model)
    explainer = shap.LinearExplainer(native, X_background, feature_names=feature_names)
    return explainer(X_test)


def _tree_explanation(fitted_model, task_type, X_background, X_test, feature_names):
    import shap

    native = _native_estimator(fitted_model)
    if task_type == "binary_classification":
        explainer = shap.TreeExplainer(
            native, X_background, model_output="probability", feature_names=feature_names
        )
    else:
        explainer = shap.TreeExplainer(native, feature_names=feature_names)
    return explainer(X_test)


def _deep_explanation(fitted_model, X_background, X_test, feature_names, shap_cfg):
    """Model-agnostic Permutation explainer for torch models.

    shap's gradient-based explainers (DeepExplainer, GradientExplainer) raise
    internal tensor-shape errors on ResidualDNN's shortcut-connection
    addition -- verified against the real ResidualBlock architecture, not
    just an additivity-check tolerance issue. Falls back to the same
    black-box approach used for TabPFN/unlisted models.
    """
    import torch

    fitted_model.eval()
    device = next(fitted_model.parameters()).device

    def predict_fn(X):
        with torch.no_grad():
            t = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
            return fitted_model(t).cpu().numpy().ravel()

    return _permutation_explanation(predict_fn, X_background, X_test, feature_names, shap_cfg)


def _tabpfn_explanation(fitted_model, X_background, X_test, feature_names, shap_cfg):
    """shapiq's KV-cache-aware imputation explainer, reduced to order-1 (plain
    Shapley) values so the result is a drop-in shap.Explanation like every
    other model family.
    """
    import shap
    from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer

    native = _native_estimator(fitted_model)
    background = _subsample(X_background, shap_cfg.get("background_size", 100))
    budget = shap_cfg.get("tabpfn_budget", 128)

    explainer = get_tabpfn_imputation_explainer(
        model=native, data=background, index="SV", max_order=1, imputer="baseline",
    )

    interaction_values = [explainer.explain(row, budget=budget) for row in X_test]
    values = np.stack([iv.get_n_order_values(1) for iv in interaction_values])
    base_values = np.array([float(iv.baseline_value) for iv in interaction_values])

    return shap.Explanation(
        values=values, base_values=base_values, data=X_test, feature_names=feature_names,
    )


def _permutation_explanation(predict_fn, X_background, X_test, feature_names, shap_cfg):
    """Generic model-agnostic Permutation explainer given predict_fn(X) -> np.ndarray.

    Used directly for any model family without a typed explainer above (every
    model/ wrapper class exposes .predict(X) -> np.ndarray on plain ndarrays),
    and via _deep_explanation for torch models.
    """
    import shap

    background = _subsample(X_background, shap_cfg.get("background_size", 100))
    masker = shap.maskers.Independent(background)
    explainer = shap.explainers.Permutation(predict_fn, masker, feature_names=feature_names)
    return explainer(X_test, max_evals=shap_cfg.get("permutation_max_evals", 500))


def _build_explanation(fitted_model, model_name, task_type, X_background, X_test, feature_names, shap_cfg):
    background_size = shap_cfg.get("background_size", 100)
    if model_name in LINEAR_MODELS:
        background = _subsample(X_background, background_size)
        return _linear_explanation(fitted_model, background, X_test, feature_names)
    if model_name in TREE_MODELS:
        background = _subsample(X_background, background_size)
        return _tree_explanation(fitted_model, task_type, background, X_test, feature_names)
    if model_name in DEEP_MODELS:
        return _deep_explanation(fitted_model, X_background, X_test, feature_names, shap_cfg)
    if model_name == "tabpfn":
        return _tabpfn_explanation(fitted_model, X_background, X_test, feature_names, shap_cfg)
    return _permutation_explanation(fitted_model.predict, X_background, X_test, feature_names, shap_cfg)


# ---------------------------------------------------------------------------
# Numeric summaries
# ---------------------------------------------------------------------------

def _values_2d(sv) -> np.ndarray:
    """Explanation values as a plain (n_test, n_features) float array.

    Binary-classification explainers can return (n_test, n_features, n_classes);
    keep the positive class so every model family reduces to the same shape.
    """
    values = np.asarray(sv.values, dtype=np.float64)
    if values.ndim == 3:
        values = values[..., -1]
    return values


def summarize_fold_shap(sv, feature_names, fold: int) -> dict:
    """Per-feature mean SHAP for one fold's test set.

    ``mean_abs_shap`` is the usual importance magnitude; ``mean_shap`` keeps the
    sign, so a feature that consistently pushes predictions down stays negative
    instead of cancelling against one that pushes up.
    """
    values = _values_2d(sv)
    return {
        "fold": fold + 1,
        "n_test": int(values.shape[0]),
        "features": [str(f) for f in feature_names],
        "mean_abs_shap": np.abs(values).mean(axis=0).tolist(),
        "mean_shap": values.mean(axis=0).tolist(),
    }


def aggregate_fold_shap(fold_summaries: list[dict]) -> dict | None:
    """Average per-fold mean SHAP values into one per-feature record.

    Each fold contributes equally (mean of the fold means, not pooled over test
    rows), so an uneven final fold cannot dominate. ``std_*`` is the spread of
    that feature's value across folds — read it as the stability of the
    importance, not as a standard error. Features are ranked by
    ``mean_abs_shap`` descending. Returns ``None`` if no fold produced values.
    """
    folds = [f for f in fold_summaries if f]
    if not folds:
        return None

    names = folds[0]["features"]
    mismatched = [f["fold"] for f in folds if f["features"] != names]
    if mismatched:
        print(f"  SHAP aggregation skipped: feature names differ in fold(s) {mismatched}")
        return None

    mean_abs = np.array([f["mean_abs_shap"] for f in folds], dtype=np.float64)  # (folds, F)
    mean_signed = np.array([f["mean_shap"] for f in folds], dtype=np.float64)
    order = np.argsort(-mean_abs.mean(axis=0), kind="stable")

    return {
        "n_folds": len(folds),
        "folds_included": [f["fold"] for f in folds],
        "n_test_per_fold": [f["n_test"] for f in folds],
        "aggregation": "mean over folds of each fold's mean over test rows",
        "features": [
            {
                "feature": names[i],
                "mean_abs_shap": float(mean_abs[:, i].mean()),
                "std_abs_shap": float(mean_abs[:, i].std(ddof=0)),
                "mean_shap": float(mean_signed[:, i].mean()),
                "std_shap": float(mean_signed[:, i].std(ddof=0)),
                "per_fold_mean_abs_shap": mean_abs[:, i].tolist(),
            }
            for i in order
        ],
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _save_summary_plot(sv, X_test, feature_names, max_display, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(sv.values, X_test, feature_names=feature_names, max_display=max_display, show=False)
    plt.gcf().savefig(out_dir / "summary_plot.png", bbox_inches="tight")
    plt.close("all")


def _save_bar_plot(sv, max_display, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.gcf().savefig(out_dir / "bar_plot.png", bbox_inches="tight")
    plt.close("all")


def _save_force_stacked(sv, out_dir: Path, max_rows: int = 50) -> None:
    """Render each test-set row's force plot and stack them vertically into one PNG.

    shap's matplotlib force-plot renderer only supports a single row at a
    time (NotImplementedError for multi-row Explanations), so this renders
    each row individually and composites the results with Pillow. Capped at
    max_rows to keep the output image a sane size for large test folds.
    """
    import io

    import matplotlib.pyplot as plt
    import shap
    from PIL import Image

    n_show = min(sv.values.shape[0], max_rows)
    images = []
    for i in range(n_show):
        shap.plots.force(sv[i], matplotlib=True, show=False)
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close("all")
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    width = max(im.width for im in images)
    total_height = sum(im.height for im in images)
    stacked = Image.new("RGB", (width, total_height), "white")
    y = 0
    for im in images:
        stacked.paste(im, (0, y))
        y += im.height
    stacked.save(out_dir / "force_stacked.png")


def _save_dependence_plots(sv, X_test, feature_names, dependence_features, dpi, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import shap

    targets = list(feature_names) if dependence_features == "all" else list(dependence_features)
    for feature in targets:
        shap.dependence_plot(feature, sv.values, X_test, feature_names=feature_names, show=False)
        safe_name = str(feature).replace("/", "_")
        plt.gcf().savefig(out_dir / f"dependence_{safe_name}.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")


def _save_waterfall_plots(sv, waterfall_observations, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import shap

    for idx in waterfall_observations:
        shap.plots.waterfall(sv[idx], show=False)
        plt.gcf().savefig(out_dir / f"waterfall_obs{idx}.png", bbox_inches="tight")
        plt.close("all")


def explain_fold(
    fitted_model,
    model_name: str,
    task_type: str,
    X_background: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    fold: int,
    experiment_name: str,
    shap_cfg: dict,
    results_dir: Path,
) -> dict:
    """Compute SHAP values for one outer fold's test set and save all requested plots.

    X_background is the fold's (already scaled/PCA'd/noised) training data;
    X_test is the outer-fold test set the values are explained on.

    Returns the fold's per-feature mean SHAP values (see ``summarize_fold_shap``)
    so the caller can average them across folds into the results JSON. Set
    ``shap.plots: false`` to compute those numbers without rendering any figure.
    """
    # Cost is per explained row — TabPFN's imputation explainer runs one
    # explain() per row at `tabpfn_budget` model calls each. When only the
    # per-feature means are wanted, a random subset of the test set estimates
    # them at a fraction of the cost. Off by default (explain every row).
    max_rows = shap_cfg.get("max_explain_rows")
    if max_rows and len(X_test) > int(max_rows):
        n_full = len(X_test)
        X_test = _subsample(X_test, int(max_rows))
        print(f"  [{experiment_name}] fold {fold + 1}: explaining a random "
              f"{len(X_test)}/{n_full} test rows (shap.max_explain_rows)")

    sv = _build_explanation(
        fitted_model, model_name, task_type, X_background, X_test, feature_names, shap_cfg,
    )
    summary = summarize_fold_shap(sv, feature_names, fold)

    tag = f"[{experiment_name}] fold {fold + 1}"
    if not shap_cfg.get("plots", True):
        print(f"  {tag}: SHAP values computed for {summary['n_test']} test rows "
              f"x {len(feature_names)} features (plots off)")
        return summary

    out_dir = Path(results_dir) / "shap" / f"fold_{fold + 1}"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_display = shap_cfg.get("max_display", 20)

    plots = [
        ("summary", lambda: _save_summary_plot(sv, X_test, feature_names, max_display, out_dir)),
        ("bar", lambda: _save_bar_plot(sv, max_display, out_dir)),
    ]
    if shap_cfg.get("force_stacked", True):
        plots.append(("stacked force", lambda: _save_force_stacked(
            sv, out_dir, max_rows=shap_cfg.get("force_stacked_max_rows", 50),
        )))
    plots.append(("dependence", lambda: _save_dependence_plots(
        sv, X_test, feature_names, shap_cfg.get("dependence_features", []),
        shap_cfg.get("dependence_dpi", 50), out_dir,
    )))
    plots.append(("waterfall", lambda: _save_waterfall_plots(
        sv, shap_cfg.get("waterfall_observations", []), out_dir,
    )))

    for plot_name, fn in plots:
        try:
            fn()
        except Exception as e:
            print(f"  {tag}: SHAP {plot_name} plot failed: {e}")

    print(f"  {tag}: SHAP plots saved to {out_dir}")
    return summary
