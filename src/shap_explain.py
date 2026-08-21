"""Per-fold shapiq explanations for the outer-fold test set.

Called from src.cv.nested_cv after a fold's final model is trained (never during
inner-fold HPO). Every model family goes through the same entry point --
``shapiq.Explainer`` -- which dispatches on the estimator: a ``TabularExplainer``
(marginal imputation) for the sklearn linear family, a ``TabPFNExplainer`` for
TabPFN. Only those two families are supported; anything else is skipped with a
message rather than explained by a fallback.

Two modes, chosen in the experiment file's ``shap:`` section:

  interactions: false  ->  index "SV",    max_order 1  (plain Shapley values)
  interactions: true   ->  index "k-SII", max_order 2  (pairwise interactions)

``index`` and ``max_order`` can be set explicitly to override either default
(e.g. ``max_order: 3`` for triples, or ``index: FSII``).

Each explained row yields one ``InteractionValues``. Those are averaged over the
rows of a fold, then over the outer folds, so an experiment ends with the mean
attribution of every feature (and every feature pair, in interaction mode).

Averaging is done twice, because it cannot keep sign and magnitude at once:

  signed        mean of the values          -> ``mean_shap`` / ``mean_value``
  magnitude     mean of their absolute      -> ``mean_abs_shap`` / ``mean_abs_value``

A feature that pushes predictions up on half the rows and down on the other half
has a signed mean near zero and a large magnitude, so the two rank features
differently -- the magnitude is the importance ranking, the signed mean is the
average direction. Both are written to ``<results_dir>/shap/`` as
``interaction_values_mean.json`` and ``interaction_values_mean_abs.json`` in
shapiq's own format (reload with
``shapiq.InteractionValues.from_json_file(path)``), and both appear per feature
in the experiment's result JSON under ``shap_mean_values``. Every plot is drawn
from whichever of the two it belongs to -- see ``render_plots``.

The beeswarm is the exception: it keeps the per-row explanations instead of an
average, so the sign and spread that both means give up stay visible -- one dot
per sample, coloured by that sample's feature value. Per fold it covers that
fold's test rows; across folds the rows are pooled, and because the outer folds
are disjoint that is exactly one explanation per sample in the dataset.

When ``cfg["shap"]["store"]`` is also true, the un-averaged per-row values
feeding the beeswarm are additionally written to
``<results_dir>/shap/fold_<k>/row_interactions.npz`` -- ``order1`` (n_rows, F),
``order2`` (n_rows, F, F) when interactions were computed, ``test_idx``
mapping each row back to its position in the outer fold's test split, and
``row_ids`` mapping it to the dataset's own row identifier when the caller has
one. Independent of ``cfg["predictions"]["enabled"]``, which only controls
whether predictions themselves are saved. See ``save_row_interactions``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# The sklearn linear family. shapiq sees the unwrapped estimator and picks its
# TabularExplainer.
LINEAR_MODELS: frozenset[str] = frozenset({
    "linear_regression", "lasso_regression", "ridge_regression", "elastic_regression",
    "bayesian_ridge_regression",
    "logistic_regression", "ridge_logistic_regression", "lasso_logistic_regression",
    "elastic_logistic_regression",
})

# shapiq's TabPFNExplainer: no imputation, the model is re-conditioned on each
# coalition's feature subset instead.
TABPFN_MODELS: frozenset[str] = frozenset({"tabpfn"})

SUPPORTED_MODELS: frozenset[str] = LINEAR_MODELS | TABPFN_MODELS

# Exhaustive computation needs 2**n_features coalitions, which is only tractable
# for small feature sets. `budget: auto` asks for the exhaustive count and lets
# this cap it -- above the cap shapiq switches to sampling and marks the result
# as estimated.
DEFAULT_MAX_BUDGET = 2048


def _native_estimator(fitted_model):
    """Unwrap a model/ wrapper class down to the library object shapiq expects.

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


def _subsample_xy(X: np.ndarray, y: np.ndarray | None, size: int, seed: int = 42):
    """Row-aligned subsample of a background set and its labels."""
    if y is None:
        return _subsample(X, size, seed), None
    if len(X) <= size:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=size, replace=False)
    return X[idx], np.asarray(y)[idx]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_index(shap_cfg: dict) -> tuple[str, int]:
    """(index, max_order) from the config. ``interactions: true`` flips the default.

    Explicit ``index``/``max_order`` always win. "SV" only exists at order 1, so
    asking for a higher order alongside it is treated as a request for k-SII --
    shapiq would otherwise reject the combination.
    """
    interactions = bool(shap_cfg.get("interactions", False))
    index = shap_cfg.get("index") or ("k-SII" if interactions else "SV")
    max_order = int(shap_cfg.get("max_order", 2 if interactions else 1))
    if index == "SV" and max_order > 1:
        index = "k-SII"
    if index != "SV" and max_order < 1:
        max_order = 1
    return index, max_order


def resolve_budget(shap_cfg: dict, n_features: int) -> int:
    """Coalitions sampled per explained row.

    ``budget: auto`` (the default) means the exhaustive 2**n_features, capped at
    ``max_budget`` so a 40-feature matrix does not ask for 2**40 model calls.
    An explicit integer is passed through untouched.
    """
    budget = shap_cfg.get("budget", "auto")
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        return int(budget)
    max_budget = int(shap_cfg.get("max_budget", DEFAULT_MAX_BUDGET))
    exhaustive = 2 ** n_features if n_features < 31 else float("inf")
    return int(min(exhaustive, max_budget))


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def _build_explainer(
    fitted_model,
    model_name: str,
    X_background: np.ndarray,
    y_background: np.ndarray | None,
    X_explain: np.ndarray,
    index: str,
    max_order: int,
    shap_cfg: dict,
):
    """One shapiq.Explainer for this fold.

    The TabPFN path needs the training labels as well: it re-conditions the model
    on each coalition instead of imputing, so it re-fits from (data, labels). It
    also takes the mean prediction over the explained rows as the empty
    prediction. Neither argument exists on the tabular path -- they would land in
    the marginal imputer's kwargs and raise -- so it gets `random_state` instead.
    """
    import shapiq
    from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq


    native = _native_estimator(fitted_model)
    background_size = int(shap_cfg.get("background_size", 100))
    bg_X, bg_y = _subsample_xy(X_background, y_background, background_size)

    if model_name in TABPFN_MODELS:
        if bg_y is None:
            raise ValueError("TabPFN explanations need the fold's training labels")
        empty_prediction = float(np.mean(native.predict(X_explain)))
        import os

        # Prevent OpenMP/MKL thread conflicts with TabPFN's PyTorch backend
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        #return shapiq.Explainer(
        #    model=native, data=bg_X, labels=bg_y,
        #    index=index, max_order=max_order,
        #    empty_prediction=empty_prediction,
        #)
        return tabpfn_shapiq.get_tabpfn_imputation_explainer(model=native, data=bg_X, index=index, max_order=max_order)

    # sample_size is how many background rows the marginal imputer draws per
    # coalition to estimate that coalition's expected prediction. shapiq defaults
    # it to 100, which resamples an already-subsampled background and leaves a
    # per-feature offset that averaging cannot remove (its error is constant
    # across rows, not noise around them). Defaulting to the whole background
    # makes the estimate exact given that background; cost scales with it, so
    # background_size is the dial for the accuracy/cost trade-off.
    return shapiq.Explainer(
        model=native, data=bg_X,
        index=index, max_order=max_order,
        random_state=int(shap_cfg.get("random_state", 0)),
        sample_size=int(shap_cfg.get("imputer_sample_size", len(bg_X))),
    )


def _interaction_values_like(template, scores: dict, baseline_value: float, estimated: bool):
    """An InteractionValues with the same shape as ``template`` but new scores."""
    import shapiq

    return shapiq.InteractionValues(
        values=dict(scores),
        index=template.index,
        max_order=template.max_order,
        min_order=template.min_order,
        n_players=template.n_players,
        baseline_value=baseline_value,
        estimated=estimated,
    )


def _mean_interaction_values(ivs: list, baseline_values: list[float]):
    """Element-wise mean of a list of InteractionValues.

    shapiq's InteractionValues supports ``+`` but not division, and two runs can
    order their value vectors differently, so this averages through
    ``dict_values`` (interaction tuple -> score) and rebuilds the object from the
    resulting mapping. Interactions missing from a run count as 0 for that run,
    which is what a sampling approximator not reaching them means.
    """
    import shapiq

    if not ivs:
        return None

    n = len(ivs)
    totals: dict[tuple[int, ...], float] = {}
    for iv in ivs:
        for key, value in iv.dict_values.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    mean_scores = {key: total / n for key, total in totals.items()}

    first = ivs[0]
    return shapiq.InteractionValues(
        values=mean_scores,
        index=first.index,
        max_order=first.max_order,
        min_order=first.min_order,
        n_players=first.n_players,
        baseline_value=float(np.mean(baseline_values)),
        estimated=any(iv.estimated for iv in ivs),
    )


def _order_1_arrays(ivs: list, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean signed, mean absolute) order-1 values over a list of InteractionValues."""
    values = np.zeros((len(ivs), n_features), dtype=np.float64)
    for i, iv in enumerate(ivs):
        values[i] = np.asarray(iv.get_n_order_values(1), dtype=np.float64)
    return values.mean(axis=0), np.abs(values).mean(axis=0)


def _mean_abs_by_interaction(ivs: list) -> dict[tuple[int, ...], float]:
    """Interaction tuple -> mean |score| over the explained rows.

    An interaction's sign flips with the instance, so the signed mean of a real
    interaction can sit near zero while every row shows a large effect. Keeping
    the magnitude alongside gives the ranking something that does not cancel --
    the same reason ``mean_abs_shap`` exists for the order-1 values.
    """
    n = len(ivs)
    totals: dict[tuple[int, ...], float] = {}
    for iv in ivs:
        for key, value in iv.dict_values.items():
            totals[key] = totals.get(key, 0.0) + abs(float(value))
    return {key: total / n for key, total in totals.items()}


def _row_interaction_arrays(
    ivs: list, n_features: int, max_order: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dense per-row arrays built from each row's ``InteractionValues``.

    ``order1`` is (n_rows, F) -- the same order-1 values ``_order_1_arrays``
    averages, kept per row instead of collapsed. ``order2`` is a symmetric
    (n_rows, F, F) matrix of pairwise interactions, built from ``dict_values``
    (the same tuple-keyed mapping ``_mean_abs_by_interaction`` reads) rather
    than a shapiq accessor, since only order-1 has one here. ``None`` when
    ``max_order < 2`` (no pairwise interactions were computed).
    """
    order1 = np.zeros((len(ivs), n_features), dtype=np.float64)
    for i, iv in enumerate(ivs):
        order1[i] = np.asarray(iv.get_n_order_values(1), dtype=np.float64)

    if max_order < 2:
        return order1, None

    order2 = np.zeros((len(ivs), n_features, n_features), dtype=np.float64)
    for i, iv in enumerate(ivs):
        for key, value in iv.dict_values.items():
            if len(key) == 2:
                a, b = key
                order2[i, a, b] = float(value)
                order2[i, b, a] = float(value)
    return order1, order2


def save_row_interactions(
    ivs: list,
    test_idx: np.ndarray,
    row_ids: np.ndarray | None,
    feature_names: list[str],
    out_dir: Path,
    max_order: int,
    tag: str,
) -> Path:
    """Persist one fold's per-row Shapley/interaction values as a dense ``.npz``.

    ``test_idx`` are the *positional* sample indices (pre-fold-split row
    numbers) these rows came from, so the array can be matched back to a
    sample after folds are stitched together -- the same role
    ``fold_assignment`` plays for predictions. ``row_ids`` are the dataset's
    own row identifiers (e.g. subject IDs) at those same positions, when the
    caller has them -- omitted from the file when not given. ``order2`` is
    only written when interactions were computed (``max_order >= 2``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    order1, order2 = _row_interaction_arrays(ivs, len(feature_names), max_order)
    baseline_values = np.array([float(iv.baseline_value) for iv in ivs], dtype=np.float64)
    path = out_dir / "row_interactions.npz"
    save_kwargs = dict(
        order1=order1,
        test_idx=np.asarray(test_idx),
        baseline_values=baseline_values,
        feature_names=np.asarray([str(f) for f in feature_names], dtype=object),
    )
    if order2 is not None:
        save_kwargs["order2"] = order2
    if row_ids is not None:
        save_kwargs["row_ids"] = np.asarray(row_ids)
    np.savez(path, **save_kwargs)
    print(f"  {tag}: per-row interaction values saved to {path}")
    return path


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
    y_background: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    row_ids: np.ndarray | None = None,
) -> dict:
    """Explain one outer fold's test rows and return the fold's mean attribution.

    X_background is the fold's (already scaled/PCA'd/noised) training data --
    background distribution for the tabular imputer, training set for TabPFN's
    re-conditioning. X_test is the outer-fold test set the values are computed on.

    The returned dict carries the fold's mean InteractionValues (mean over test
    rows) plus the order-1 summary, for ``aggregate_fold_shap`` to average across
    folds. Returns ``{}`` for a model family shapiq is not being used on here.

    When ``shap_cfg["store"]`` is true, the per-row values computed here (which
    already exist in memory to feed the beeswarm plot) are additionally
    persisted to ``<results_dir>/shap/fold_<k>/row_interactions.npz`` via
    ``save_row_interactions`` -- independent of anything outside ``shap_cfg``.
    ``test_idx``/``row_ids`` are only used for that file (positional index and
    dataset row identifier, respectively); omit them to skip the save even
    when ``store`` is true.
    """
    tag = f"[{experiment_name}] fold {fold + 1}"
    if model_name not in SUPPORTED_MODELS:
        print(f"  {tag}: shapiq explanations cover the linear family and TabPFN — "
              f"skipping {model_name}")
        return {}

    # Cost is per explained row -- one explain() of `budget` model calls each.
    # When only the means are wanted, a random subset of the test set estimates
    # them at a fraction of the cost. Off by default (explain every row).
    max_rows = shap_cfg.get("max_explain_rows")
    if max_rows and len(X_test) > int(max_rows):
        n_full = len(X_test)
        rng = np.random.default_rng(42)
        subsample_idx = rng.choice(n_full, size=int(max_rows), replace=False)
        X_test = X_test[subsample_idx]
        if test_idx is not None:
            test_idx = np.asarray(test_idx)[subsample_idx]
        if row_ids is not None:
            row_ids = np.asarray(row_ids)[subsample_idx]
        print(f"  {tag}: explaining a random {len(X_test)}/{n_full} test rows "
              f"(shap.max_explain_rows)")

    index, max_order = resolve_index(shap_cfg)
    budget = resolve_budget(shap_cfg, len(feature_names))
    random_state = int(shap_cfg.get("random_state", 0))

    explainer = _build_explainer(
        fitted_model, model_name, X_background, y_background, X_test,
        index, max_order, shap_cfg,
    )
    print(f"  {tag}: {type(explainer).__name__} index={index} max_order={max_order} "
          f"budget={budget} on {len(X_test)} test rows x {len(feature_names)} features")

    # explain_X takes the whole block of rows at once and can spread them over
    # cores via joblib. n_jobs stays None (serial) by default -- TabPFN holds a
    # torch model whose workers would each need their own copy.
    ivs = explainer.explain_X(
        X_test,
        budget=budget,
        random_state=random_state,
        n_jobs=shap_cfg.get("n_jobs"),
        verbose=bool(shap_cfg.get("verbose", False)),
    )
    fold_iv = _mean_interaction_values(ivs, [float(iv.baseline_value) for iv in ivs])
    mean_signed, mean_abs = _order_1_arrays(ivs, len(feature_names))
    abs_by_key = _mean_abs_by_interaction(ivs)

    if bool(shap_cfg.get("store", False)) and test_idx is not None:
        row_dir = Path(results_dir) / "shap" / f"fold_{fold + 1}"
        save_row_interactions(ivs, test_idx, row_ids, feature_names, row_dir, max_order, tag)

    summary = {
        "fold": fold + 1,
        "n_test": int(len(X_test)),
        "features": [str(f) for f in feature_names],
        "index": index,
        "max_order": max_order,
        "budget": budget,
        "estimated": bool(fold_iv.estimated),
        "baseline_value": float(fold_iv.baseline_value),
        "mean_abs_shap": mean_abs.tolist(),
        "mean_shap": mean_signed.tolist(),
        # Interaction tuple -> score over this fold's test rows, signed mean and
        # mean magnitude. Not JSON (tuple keys); consumed by
        # aggregate_fold_shap, which serialises them under feature names.
        "interaction_values": {k: float(v) for k, v in fold_iv.dict_values.items()},
        "interaction_abs_values": abs_by_key,
        # The un-averaged per-row explanations and the rows they belong to, for
        # the beeswarm -- the one plot that needs the distribution rather than a
        # summary of it. In memory only: the caller aggregates them and never
        # serialises them (they are neither JSON-safe nor small).
        "row_interaction_values": ivs,
        "row_data": np.asarray(X_test),
    }

    if shap_cfg.get("plots", True):
        out_dir = Path(results_dir) / "shap" / f"fold_{fold + 1}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_iv_abs = _interaction_values_like(
            fold_iv, abs_by_key, float(fold_iv.baseline_value), bool(fold_iv.estimated),
        )
        render_plots(fold_iv, fold_iv_abs, feature_names, out_dir, shap_cfg, tag,
                     row_ivs=ivs, row_data=X_test)
        print(f"  {tag}: shapiq plots saved to {out_dir}")

    return summary


# ---------------------------------------------------------------------------
# Aggregation across the outer folds
# ---------------------------------------------------------------------------

def aggregate_fold_shap(
    fold_summaries: list[dict],
    results_dir: Path | str | None = None,
    experiment_name: str = "",
    shap_cfg: dict | None = None,
) -> dict | None:
    """Average the per-fold mean attributions into one record for the experiment.

    Each fold contributes equally (mean of the fold means, not pooled over test
    rows), so an uneven final fold cannot dominate. ``std_*`` is the spread of
    that value across folds -- read it as the stability of the attribution, not
    as a standard error.

    When ``results_dir`` is given, the averaged InteractionValues is also written
    to ``<results_dir>/shap/interaction_values_mean.json`` in shapiq's own format
    and the aggregate plots are rendered next to it. Returns ``None`` if no fold
    produced values.
    """
    shap_cfg = shap_cfg or {}
    folds = [f for f in fold_summaries if f]
    if not folds:
        return None

    names = folds[0]["features"]
    mismatched = [f["fold"] for f in folds if f["features"] != names]
    if mismatched:
        print(f"  shapiq aggregation skipped: feature names differ in fold(s) {mismatched}")
        return None

    mean_abs = np.array([f["mean_abs_shap"] for f in folds], dtype=np.float64)   # (folds, F)
    mean_signed = np.array([f["mean_shap"] for f in folds], dtype=np.float64)
    order = np.argsort(-mean_abs.mean(axis=0), kind="stable")

    # Interaction tuple -> its value in each fold (0.0 where a fold never
    # reached that interaction).
    keys = sorted({k for f in folds for k in f["interaction_values"]}, key=lambda k: (len(k), k))
    per_fold = {k: [float(f["interaction_values"].get(k, 0.0)) for f in folds] for k in keys}
    per_fold_abs = {
        k: [float(f.get("interaction_abs_values", {}).get(k, 0.0)) for f in folds] for k in keys
    }

    max_order = max(int(f["max_order"]) for f in folds)
    result = {
        "n_folds": len(folds),
        "folds_included": [f["fold"] for f in folds],
        "n_test_per_fold": [f["n_test"] for f in folds],
        "index": folds[0]["index"],
        "max_order": max_order,
        "budget": folds[0]["budget"],
        "estimated": any(f["estimated"] for f in folds),
        "baseline_value": float(np.mean([f["baseline_value"] for f in folds])),
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

    if max_order > 1:
        # Order >= 2 only; the order-1 values are already in "features" above.
        # Ranked by mean magnitude, which (unlike the signed mean) does not
        # cancel across instances whose interaction points the other way.
        higher = [k for k in keys if len(k) > 1]
        higher.sort(key=lambda k: -float(np.mean(per_fold_abs[k])))
        result["interactions"] = [
            {
                "features": [names[i] for i in k],
                "order": len(k),
                "mean_abs_value": float(np.mean(per_fold_abs[k])),
                "std_abs_value": float(np.std(per_fold_abs[k], ddof=0)),
                "mean_value": float(np.mean(per_fold[k])),
                "std_value": float(np.std(per_fold[k], ddof=0)),
                "per_fold_value": per_fold[k],
            }
            for k in higher
        ]

    # Two objects: the signed mean (what "the average across the folds" means)
    # and the mean magnitude (what the importance plots and the JSON ranking
    # use). Averaging cannot preserve both -- see render_plots.
    mean_iv = _rebuild_mean_iv(folds, per_fold, len(names), result["baseline_value"])
    mean_iv_abs = _rebuild_mean_iv(folds, per_fold_abs, len(names), result["baseline_value"])
    if results_dir is not None and mean_iv is not None:
        out_dir = Path(results_dir) / "shap"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for suffix, obj, what in (
            ("mean", mean_iv, "mean"),
            ("mean_abs", mean_iv_abs, "mean absolute"),
        ):
            if obj is None:
                continue
            path = out_dir / f"interaction_values_{suffix}.json"
            try:
                obj.to_json_file(
                    path,
                    desc=f"{experiment_name}: {what} interaction values over "
                         f"{len(folds)} outer folds ({result['aggregation']})",
                )
                saved[suffix] = str(path)
                print(f"  [{experiment_name}] {what} interaction values saved to {path}")
            except Exception as e:
                print(f"  [{experiment_name}] saving {what} interaction values failed: {e}")
        if saved:
            result["interaction_values_files"] = saved
        if shap_cfg.get("plots", True) and mean_iv_abs is not None:
            # The folds' test sets are disjoint, so pooling their rows gives one
            # explanation per sample in the dataset -- the out-of-fold beeswarm.
            row_ivs = [iv for f in folds for iv in f.get("row_interaction_values", [])]
            row_data = [f["row_data"] for f in folds if f.get("row_data") is not None]
            row_data = np.vstack(row_data) if row_data else None
            render_plots(mean_iv, mean_iv_abs, names, out_dir, shap_cfg,
                         f"[{experiment_name}] mean",
                         row_ivs=row_ivs or None, row_data=row_data)
            print(f"  [{experiment_name}] mean shapiq plots saved to {out_dir}")

    return result


def _rebuild_mean_iv(folds: list[dict], per_fold: dict, n_features: int, baseline: float):
    """The across-fold mean of a per-fold {interaction: value} mapping."""
    import shapiq

    try:
        return shapiq.InteractionValues(
            values={k: float(np.mean(v)) for k, v in per_fold.items()},
            index=folds[0]["index"],
            max_order=max(int(f["max_order"]) for f in folds),
            min_order=0,
            n_players=n_features,
            baseline_value=baseline,
            estimated=any(f["estimated"] for f in folds),
        )
    except Exception as e:
        print(f"  mean InteractionValues could not be rebuilt: {e}")
        return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_bar_plot(iv_abs, feature_names, max_display: int, out_dir: Path) -> None:
    """Mean |attribution| per feature -- shapiq has no single-object bar plot.

    Drawn from the magnitude object, so the bars are exactly the
    ``mean_abs_shap`` numbers in the result JSON, in the same order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(iv_abs.get_n_order_values(1), dtype=np.float64)
    order = np.argsort(-values)[:max_display][::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(order))), dpi=150)
    ax.barh([feature_names[i] for i in order], values[order], color="#2a78d6")
    ax.set_xlabel("mean |attribution|")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_plot.png", bbox_inches="tight")
    plt.close(fig)


def _save_beeswarm_plot(row_ivs, row_data, feature_names, max_display: int,
                        shap_cfg: dict, out_dir: Path) -> None:
    """One dot per explained sample, per feature -- sign and spread intact.

    Feature names are passed through unabbreviated so the rows read the same as
    the result JSON. Very large point clouds are thinned first (the plot stops
    being readable long before it stops being expensive); ``beeswarm_max_rows``
    lifts or lowers that cap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shapiq

    cap = int(shap_cfg.get("beeswarm_max_rows", 5000))
    data = np.asarray(row_data)
    ivs = list(row_ivs)
    if len(ivs) > cap:
        idx = np.random.default_rng(42).choice(len(ivs), size=cap, replace=False)
        idx.sort()
        ivs = [ivs[i] for i in idx]
        data = data[idx]

    shapiq.plot.beeswarm_plot(
        ivs, data,
        feature_names=list(feature_names),
        max_display=max_display,
        abbreviate=bool(shap_cfg.get("abbreviate_names", False)),
        show=False,
    )
    plt.gcf().savefig(out_dir / "beeswarm_plot.png", bbox_inches="tight", dpi=150)
    plt.close("all")


def render_plots(iv, iv_abs, feature_names, out_dir: Path, shap_cfg: dict, tag: str,
                 row_ivs: list | None = None, row_data=None) -> None:
    """Render the plots for one attribution, each from the object it belongs to.

    Two averaged objects, because averaging over rows and folds can only preserve
    one of sign and magnitude:

      iv      signed mean  -> ``mean_shap`` / ``mean_value`` in the result JSON.
                             Feeds force and waterfall, which decompose the mean
                             prediction and need the signs to add up.
      iv_abs  mean |value| -> ``mean_abs_shap`` / ``mean_abs_value``. Feeds the
                             importance plots (bar, network, SI graph, upset,
                             stacked bar), where a signed mean would cancel: a
                             feature pushing predictions up on half the rows and
                             down on the other half averages to ~0 while every
                             row shows a large effect.

    ``row_ivs``/``row_data`` are the un-averaged per-row explanations and their
    rows. They feed the beeswarm, which is the plot that answers what both
    averages give up: one dot per sample, positioned by its signed attribution
    and coloured by its feature value, so the sign and the spread stay visible.

    Interaction plots are skipped when the object holds only order-1 values.
    Each plot is independent -- one failing never stops the rest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_display = int(shap_cfg.get("max_display", 20))
    plots: list[tuple[str, callable]] = [
        ("bar", lambda: _save_bar_plot(iv_abs, feature_names, max_display, out_dir)),
        ("force", lambda: iv.plot_force(feature_names=feature_names, show=False)),
        ("waterfall", lambda: iv.plot_waterfall(feature_names=feature_names, show=False)),
    ]
    if row_ivs is not None and row_data is not None and len(row_ivs) == len(row_data):
        plots.append(("beeswarm", lambda: _save_beeswarm_plot(
            row_ivs, row_data, feature_names, max_display, shap_cfg, out_dir)))
    if iv.max_order > 1:
        plots += [
            ("network", lambda: iv_abs.plot_network(feature_names=feature_names, show=False)),
            ("si_graph", lambda: iv_abs.plot_si_graph(feature_names=feature_names, show=False)),
            ("upset", lambda: iv_abs.plot_upset(feature_names=feature_names, show=False)),
            ("stacked_bar", lambda: iv_abs.plot_stacked_bar(feature_names=feature_names, show=False)),
        ]

    writes_own_file = {"bar", "beeswarm"}
    for name, fn in plots:
        try:
            fn()
            if name not in writes_own_file:
                plt.gcf().savefig(out_dir / f"{name}_plot.png", bbox_inches="tight")
                plt.close("all")
        except Exception as e:
            print(f"  {tag}: shapiq {name} plot failed: {e}")
            plt.close("all")
