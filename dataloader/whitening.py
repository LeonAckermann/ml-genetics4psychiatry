"""ZCA whitening of a target + its feature set by a phenotype covariance
matrix (Sigma), for use inside ``dataloader.preprocess.sample``.

Self-contained on purpose: no dependency on the standalone ``whitening/``
diagnostic module (pre-flight report, plots, sign-consistency checks, etc.)
at the repo root — this file only carries the small subset of that module's
logic actually needed to fit and apply a production whitener, so this package
doesn't reach outside itself.

Deliberately whitens the *raw* Sigma, not a correlation-rescaled version. For
LDSC/JASS z-scores the diagonal (``1 + a_k``) is each trait's actual null
variance under the model — a trait with more sample-overlap-driven inflation
genuinely has higher variance in the data being whitened, so ZCA on the raw
covariance correctly downweights it as part of the same operation that
removes cross-trait correlation. Rescaling to correlation form first would
discard that and understate the correction for the more-inflated traits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Strings that appear in this project's matrices where a value is absent.
NULL_STRINGS = ["", "NA", "NaN", "nan", "Null", "NULL", "null", "None", "."]


# ---------------------------------------------------------------------------
# Sigma loading
# ---------------------------------------------------------------------------

def load_matrix(
    path: str | Path,
    labels_path: str | Path | None = None,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load a labelled square matrix (``.csv``/``.tsv``/``.parquet``/``.npy``).

    ``.csv``/``.tsv``/``.parquet`` take the first column as row labels and the
    header as column labels (the layout of ``official_intercept_matrix.csv``).
    ``.npy`` is unlabelled; ``labels_path`` must point at a text file with one
    label per line.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        A = np.load(path)
        if labels_path is None:
            raise ValueError(f"{path} is unlabelled .npy — supply labels_path.")
        labels = [ln.strip() for ln in Path(labels_path).read_text().splitlines() if ln.strip()]
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df.set_index(df.columns[0])
        A, labels = df.to_numpy(dtype=float), [str(c) for c in df.columns]
        _check_square_labels([str(i) for i in df.index], labels, path)
    else:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0, na_values=NULL_STRINGS)
        A = df.to_numpy(dtype=float)
        labels = [str(c) for c in df.columns]
        _check_square_labels([str(i) for i in df.index], labels, path)

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{path}: expected a square matrix, got shape {A.shape}")
    if len(labels) != A.shape[0]:
        raise ValueError(f"{path}: {len(labels)} labels for a {A.shape} matrix")

    if columns is not None:
        idx = _label_index(labels, columns, str(path))
        A = A[np.ix_(idx, idx)]
        labels = list(columns)

    return A, labels


def _check_square_labels(row_labels: list[str], col_labels: list[str], path: Path) -> None:
    if row_labels != col_labels:
        n_bad = sum(r != c for r, c in zip(row_labels, col_labels))
        raise ValueError(
            f"{path}: row labels do not match column labels ({n_bad} positions "
            "differ). The matrix is not indexed consistently; fix before whitening."
        )


def _label_index(available: list[str], wanted: list[str], where: str) -> list[int]:
    pos = {lab: i for i, lab in enumerate(available)}
    missing = [w for w in wanted if w not in pos]
    if missing:
        raise KeyError(
            f"{where}: {len(missing)} requested trait(s) absent: "
            f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}"
        )
    return [pos[w] for w in wanted]


# ---------------------------------------------------------------------------
# Regularisation + whitening transform
# ---------------------------------------------------------------------------

def regularize(
    S: np.ndarray,
    method: str = "ridge",
    alpha: float = 0.0,
    renormalize: bool | None = None,
) -> np.ndarray:
    """Return a regularised copy of ``S``.

    ``ridge``  ``S + alpha * I``, optionally renormalised back to unit
               diagonal if ``S`` already had one (it doesn't for a raw
               LDSC/JASS intercept matrix, so this is a no-op here).
    ``floor``  ``lambda -> max(lambda, alpha * lambda_max)``.
    ``higham`` Nearest positive-definite matrix (diagonal held fixed).
    ``none``   Passthrough.
    """
    if method == "none" or alpha == 0.0 and method == "ridge":
        return S.copy()

    if method == "ridge":
        out = S + alpha * np.eye(S.shape[0])
        if renormalize is None:
            renormalize = bool(np.allclose(np.diag(S), 1.0))
        return out / (1.0 + alpha) if renormalize else out

    if method == "floor":
        lam, V = np.linalg.eigh(S)
        floor = alpha * float(lam[-1])
        return (V * np.maximum(lam, floor)) @ V.T

    if method == "higham":
        return nearest_pd(S, eps=alpha * float(np.linalg.eigvalsh(S)[-1]))

    raise ValueError(f"unknown regularisation method: {method!r}")


def nearest_pd(S: np.ndarray, eps: float = 0.0, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Higham's nearest PD matrix with the diagonal held fixed."""
    d = np.diag(S).copy()
    Y = S.copy()
    dS = np.zeros_like(S)
    for _ in range(max_iter):
        R = Y - dS
        lam, V = np.linalg.eigh(0.5 * (R + R.T))
        X = (V * np.maximum(lam, eps)) @ V.T
        dS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, d)
        if np.linalg.norm(Y - X, "fro") <= tol * max(np.linalg.norm(Y, "fro"), 1.0):
            break
    lam = np.linalg.eigvalsh(Y)
    if lam[0] <= 0:
        Y = Y + (eps - lam[0] + 1e-12) * np.eye(len(Y))
    return 0.5 * (Y + Y.T)


def build_whitener(S: np.ndarray, transform: str = "zca") -> np.ndarray:
    """Return ``W`` with ``Cov(W z) = I`` when ``Cov(z) = S``."""
    S = 0.5 * (S + S.T)
    if transform == "zca":
        lam, V = np.linalg.eigh(S)
        _require_pd(lam)
        return (V * lam ** -0.5) @ V.T
    if transform == "zca-cor":
        d = np.sqrt(np.diag(S))
        R = S / np.outer(d, d)
        lam, V = np.linalg.eigh(R)
        _require_pd(lam)
        return ((V * lam ** -0.5) @ V.T) / d[None, :]
    if transform == "pca":
        lam, V = np.linalg.eigh(S)
        _require_pd(lam)
        return (V * lam ** -0.5).T
    if transform == "cholesky":
        L = np.linalg.cholesky(S)
        return np.linalg.inv(L)
    raise ValueError(f"unknown transform: {transform!r}")


def _require_pd(lam: np.ndarray) -> None:
    if lam[0] <= 0:
        raise np.linalg.LinAlgError(
            f"cannot whiten: lambda_min = {lam[0]:.4g} <= 0. Regularise first."
        )


# ---------------------------------------------------------------------------
# Null calibration + regularisation sweep
# ---------------------------------------------------------------------------

def simulate_null(S: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Draw z ~ N(0, S), n samples x K traits."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(S.shape[0]), S, size=n, method="eigh")


def whiten_rows(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    return Z @ W.T


def calibration_metrics(w: np.ndarray) -> dict:
    K = w.shape[1]
    cov = np.cov(w, rowvar=False)
    off = cov[~np.eye(K, dtype=bool)]
    quad = np.einsum("ij,ij->i", w, w)

    quantiles = [1e-1, 1e-2, 1e-3, 1e-4]
    n = len(quad)
    tail = {}
    for q in quantiles:
        if n * q < 5:
            continue
        emp = float(np.quantile(quad, 1 - q))
        ref = float(stats.chi2.ppf(1 - q, df=K))
        tail[q] = {"empirical": emp, "chi2_reference": ref, "ratio": emp / ref if ref else np.nan}

    return {
        "K": K, "n": n,
        "diag_mean": float(np.mean(np.diag(cov))),
        "diag_std": float(np.std(np.diag(cov))),
        "offdiag_mean_abs": float(np.mean(np.abs(off))),
        "offdiag_max_abs": float(np.max(np.abs(off))),
        "mean_quad_form": float(np.mean(quad)),
        "expected_quad_form": float(K),
        "quad_form_inflation": float(np.mean(quad) / K),
        "tail": tail,
    }


def sweep_regularization(
    S: np.ndarray,
    alphas: list[float],
    method: str = "ridge",
    transform: str = "zca",
    n_null: int = 20000,
    seed: int = 0,
    tail_ratio_tol: float = 2.0,
    sample_from: np.ndarray | None = None,
) -> dict:
    """Evaluate calibration at each alpha; recommend the smallest that controls the tail.

    ``sample_from`` lets the caller supply an already-regularised, guaranteed-PD
    matrix to draw the shared null z's from when ``S`` itself is not PD.
    """
    z = simulate_null(sample_from if sample_from is not None else S, n_null, seed=seed)
    rows = []
    for alpha in alphas:
        try:
            S_reg = regularize(S, method=method, alpha=alpha)
            lam_min = float(np.linalg.eigvalsh(S_reg)[0])
            if lam_min <= 0:
                rows.append({"alpha": alpha, "feasible": False, "lambda_min": lam_min})
                continue
            W = build_whitener(S_reg, transform=transform)
            w = whiten_rows(z, W)
            m = calibration_metrics(w)
            tail_ok = all(
                (1 / tail_ratio_tol) <= v["ratio"] <= tail_ratio_tol for v in m["tail"].values()
            )
            rows.append({
                "alpha": alpha, "feasible": True, "lambda_min": lam_min,
                "condition_number": float(np.linalg.eigvalsh(S_reg)[-1] / lam_min),
                "quad_form_inflation": m["quad_form_inflation"],
                "offdiag_mean_abs": m["offdiag_mean_abs"],
                "tail_ok": tail_ok, "tail": m["tail"],
            })
        except np.linalg.LinAlgError as e:
            rows.append({"alpha": alpha, "feasible": False, "error": str(e)})

    feasible = [r for r in rows if r.get("feasible") and r.get("tail_ok")]
    recommended = min(feasible, key=lambda r: r["alpha"])["alpha"] if feasible else None

    return {
        "method": method, "transform": transform, "alphas": alphas, "results": rows,
        "recommended_alpha": recommended,
    }


# ---------------------------------------------------------------------------
# Fit + apply
# ---------------------------------------------------------------------------

def _condition_number(eig: np.ndarray) -> float:
    """lambda_max/lambda_min from a sorted-ascending eigenvalue array; inf if not PD."""
    lam_min, lam_max = float(eig[0]), float(eig[-1])
    return float(lam_max / lam_min) if lam_min > 0 else float("inf")


def _format_kappa(kappa: float) -> str:
    return f"{kappa:.6g}" if np.isfinite(kappa) else "inf (not PD)"


@dataclass
class WhiteningFit:
    target: str
    features: list[str]              # order matches W's rows/cols after the target
    dropped_features: list[str]      # requested features absent from Sigma
    alpha: float
    method: str
    lambda_min_raw: float
    lambda_min_regularized: float
    kappa_full: float                 # condition number of the full loaded Sigma
    kappa_submatrix: float            # condition number of the [target]+features submatrix, pre-regularization
    transform_method: str             # "zca" / "zca-cor" / "pca" / "cholesky" -- passed to build_whitener
    W: np.ndarray                    # (1+F, 1+F) whitener; index 0 = target
    sweep: dict = field(default_factory=dict)


def fit_whitener(
    sigma_path: str,
    target: str,
    feature_names: list[str],
    method: str = "ridge",
    n_null: int = 5000,
    tail_ratio_tol: float = 2.0,
    seed: int = 0,
    labels_path: str | None = None,
    search_alpha: bool = False,
    transform_method: str = "zca",
) -> WhiteningFit:
    """Build a ZCA whitener for ``[target] + feature_names`` from Sigma.

    ``target`` is Sigma's own label for the target trait (e.g. the ``illness``
    passed to ``sample`` — NOT the dataframe's renamed ``"Z"`` column; the
    caller maps between the two).

    ``search_alpha`` (default ``True``) auto-regularizes via null-tail
    calibration if the ``[target]+features`` submatrix isn't already PD (see
    ``_auto_regularize``). Set it to ``False`` to skip that search entirely
    and build ``W`` directly from the submatrix — appropriate when
    ``sigma_path`` already points at a matrix pre-regularized to be PD as a
    whole (e.g. via ``regularize_to_condition_number`` / ``python -m
    dataloader.whitening``): any principal submatrix of a PD matrix is itself
    PD, so once the full matrix is regularized, every ``[target]+features``
    slice pulled from it is guaranteed PD too — no per-target search needed.
    A PD check still runs as a safety net; it raises rather than silently
    building a bad whitener if that assumption turns out to be wrong.

    Raises ``KeyError`` if ``target`` is not one of Sigma's traits. Features
    absent from Sigma are dropped with a warning instead of failing the whole
    run; see ``dropped_features`` on the result.
    """
    S_full, labels = load_matrix(sigma_path, labels_path=labels_path)
    pos = {lab: i for i, lab in enumerate(labels)}

    kappa_full = _condition_number(np.linalg.eigvalsh(0.5 * (S_full + S_full.T)))
    print(f"  whitening: loaded Sigma ({sigma_path}), K={len(labels)}, "
          f"condition number={_format_kappa(kappa_full)}")

    if target not in pos:
        raise KeyError(
            f"Target {target!r} not found among Sigma's {len(labels)} traits "
            f"in {sigma_path} — cannot whiten without the target's own "
            "row/column. Check that the trait name matches Sigma's labels."
        )

    kept = [f for f in feature_names if f in pos]
    dropped = [f for f in feature_names if f not in pos]
    if dropped:
        print(f"  whitening: {len(dropped)} feature(s) not found in Sigma, "
              f"dropped from whitening: {', '.join(dropped[:20])}"
              + (" ..." if len(dropped) > 20 else ""))
    if not kept:
        raise ValueError(
            f"None of the {len(feature_names)} feature(s) were found in Sigma "
            f"({sigma_path}) — nothing left to whiten."
        )

    order = [target] + kept
    idx = [pos[o] for o in order]
    S = S_full[np.ix_(idx, idx)]
    S = 0.5 * (S + S.T)

    eig = np.linalg.eigvalsh(S)
    lam_min_raw, lam_max = float(eig[0]), float(eig[-1])
    already_pd = lam_min_raw > 1e-8 * lam_max
    kappa_submatrix = _condition_number(eig)
    print(f"  whitening: [{target}]+{len(kept)} feature(s) submatrix "
          f"(K={len(order)}, present in the dataset), "
          f"condition number={_format_kappa(kappa_submatrix)}")

    if not search_alpha:
        if not already_pd:
            raise np.linalg.LinAlgError(
                f"search_alpha=False but the [{target}]+features submatrix is "
                f"not positive definite (lambda_min={lam_min_raw:.4g}). "
                f"sigma_path ({sigma_path}) needs to already be PD as a whole "
                "for this to be safe — pre-regularize it first with "
                "dataloader.whitening.regularize_to_condition_number (or "
                "`python -m dataloader.whitening --out ...`), or set "
                "search_alpha: true to auto-regularize per target instead."
            )
        alpha, S_reg = 0.0, S
        sweep = {"skipped": "search_alpha=False -- Sigma assumed already PD"}
    elif already_pd:
        alpha, S_reg = 0.0, S
        sweep = {"skipped": "already positive definite", "lambda_min": lam_min_raw}
    else:
        alpha, S_reg, sweep = _auto_regularize(
            S, method=method, n_null=n_null, tail_ratio_tol=tail_ratio_tol, seed=seed,
        )

    lam_min_reg = float(np.linalg.eigvalsh(S_reg)[0])
    W = build_whitener(S_reg, transform=transform_method)

    return WhiteningFit(
        target=target, features=kept, dropped_features=dropped,
        alpha=alpha, method=method,
        lambda_min_raw=lam_min_raw, lambda_min_regularized=lam_min_reg,
        kappa_full=kappa_full, kappa_submatrix=kappa_submatrix,
        transform_method=transform_method,
        W=W, sweep=sweep,
    )


def _auto_regularize(
    S: np.ndarray,
    method: str,
    n_null: int,
    tail_ratio_tol: float,
    seed: int,
    max_rounds: int = 4,
) -> tuple[float, np.ndarray, dict]:
    """Adaptive grid search for the smallest alpha that reaches PD-ness and
    passes null-tail calibration — no fixed grid, since Sigma's submatrix (and
    hence the alpha it needs) differs for every target phenotype.

    ``alpha`` means something different per method: for ``ridge`` it's an
    absolute eigenvalue shift (``S + alpha*I``); for ``floor``/``higham`` it's
    a *fraction of lambda_max* (``floor = alpha * lambda_max``). The search
    grid is built from a common target — "push lambda_min up to 1% of
    lambda_max" — then converted into whatever alpha each method needs to
    reach that same target, so the grid means the same thing regardless of
    which method was requested.
    """
    eig = np.linalg.eigvalsh(S)
    lam_min, lam_max = float(eig[0]), float(eig[-1])
    target_lambda_min = 1e-2 * lam_max
    if method == "ridge":
        floor = max(0.0, target_lambda_min - lam_min)
    else:  # floor, higham: alpha is a fraction of lambda_max, not an absolute shift
        floor = max(0.0, target_lambda_min / lam_max)

    for _ in range(max_rounds):
        alphas = sorted({round(floor * m, 12) for m in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)})
        sample_from = regularize(S, method=method, alpha=floor)
        result = sweep_regularization(
            S, alphas=alphas, method=method, transform="zca",
            n_null=n_null, seed=seed, tail_ratio_tol=tail_ratio_tol,
            sample_from=sample_from,
        )
        if result["recommended_alpha"] is not None:
            alpha = result["recommended_alpha"]
            return alpha, regularize(S, method=method, alpha=alpha), result
        floor = floor * 8

    raise RuntimeError(
        f"No alpha up to {floor:.4g} achieved null-tail calibration "
        f"(tail_ratio_tol={tail_ratio_tol}) after {max_rounds} widening rounds."
    )


def apply_whitening(df: pd.DataFrame, fit: WhiteningFit, target_col: str = "Z") -> tuple[pd.DataFrame, dict]:
    """Whiten ``df``'s ``[target_col] + fit.features`` columns.

    ZCA is a matrix multiply, so it needs complete rows: any row with a
    missing value across the whitened block is dropped first (reported in the
    returned stats) rather than whitened partially or left un-whitened next
    to whitened rows.
    """
    cols = [target_col] + fit.features
    n_before = len(df)

    block = df[cols].apply(pd.to_numeric, errors="coerce")
    complete = block.notna().all(axis=1)
    n_dropped = int((~complete).sum())

    # errors="coerce" turns a non-numeric cell into NaN, so a single
    # non-numeric column (a stray string, a mis-parsed null token) makes every
    # row incomplete and would otherwise return an empty frame with no error --
    # the caller only finds out much later, as a confusing downstream failure
    # on zero rows. Name the offending columns instead.
    if n_before and not complete.any():
        offenders = [c for c in cols if not block[c].notna().any()]
        raise ValueError(
            f"whitening dropped all {n_before} rows: no row is complete across "
            f"the whitened block. Column(s) entirely non-numeric after coercion: "
            f"{offenders or 'none -- missingness is spread across columns'}. "
            "Check the target/feature columns' dtypes and null representation."
        )

    df = df.loc[complete].reset_index(drop=True)
    X = block.loc[complete].reset_index(drop=True).to_numpy(dtype=np.float64)
    df[cols] = X @ fit.W.T

    stats_ = {
        "target": fit.target, "target_col": target_col,
        "n_features": len(fit.features), "dropped_features": fit.dropped_features,
        "method": fit.method, "alpha": fit.alpha,
        "lambda_min_raw": fit.lambda_min_raw, "lambda_min_regularized": fit.lambda_min_regularized,
        "kappa_full": fit.kappa_full, "kappa_submatrix": fit.kappa_submatrix,
        "transform_method": fit.transform_method,
        "sweep": fit.sweep,
        "n_rows_before": n_before, "n_rows_after": int(len(df)),
        "n_rows_dropped_incomplete": n_dropped,
    }
    return df, stats_


# ---------------------------------------------------------------------------
# Condition-number regularisation (no calibration, no simulation — just PD +
# a bounded kappa)
# ---------------------------------------------------------------------------

def regularize_to_condition_number(
    S: np.ndarray,
    target_kappa: float = 100.0,
    method: str = "ridge",
    bisect_tol: float = 1e-6,
    max_expand: int = 60,
    max_bisect: int = 60,
) -> tuple[float, np.ndarray, dict]:
    """Smallest ``alpha`` that makes ``S`` PD with ``kappa <= target_kappa``.

    No null simulation, no calibration check — just PD-ness and the spectral
    condition number, which is all that's asked for here.

    For ridge (the default), every eigenvalue of ``S + alpha*I`` shifts by
    the same additive ``alpha``, so ``kappa(alpha) = (lambda_max + alpha) /
    (lambda_min + alpha)`` decreases monotonically and strictly as ``alpha``
    grows (it has a closed form, in fact, but this loops instead so the same
    code also works unchanged for ``floor``/``higham``, whose eigenvalue
    maps aren't as simple to invert by hand). Monotonicity is what makes a
    loop guaranteed to work here: geometrically expand ``alpha`` until
    ``kappa`` first drops to or below the target (bracketing the answer),
    then bisect down to the smallest ``alpha`` in that bracket that still
    clears it.
    """
    if target_kappa <= 1.0:
        raise ValueError(f"target_kappa must be > 1, got {target_kappa}")

    def probe(alpha: float):
        S_reg = regularize(S, method=method, alpha=alpha)
        eig = np.linalg.eigvalsh(S_reg)
        lam_min, lam_max = float(eig[0]), float(eig[-1])
        kappa = float(lam_max / lam_min) if lam_min > 0 else float("inf")
        return kappa, S_reg, lam_min, lam_max

    kappa0, S0, lam_min0, lam_max0 = probe(0.0)
    if kappa0 <= target_kappa:
        return 0.0, S0, {
            "kappa": kappa0, "lambda_min": lam_min0, "lambda_max": lam_max0,
            "n_expand_steps": 0, "n_bisect_steps": 0,
        }

    # Expand geometrically (scaled to S's own eigenvalue magnitude) until
    # kappa first drops to or below target -- brackets the answer between
    # the last failing alpha and this first passing one.
    alpha_start = max(1e-6, 1e-6 * lam_max0)
    alpha_lo, alpha_hi = 0.0, alpha_start
    n_expand = 0
    for n_expand in range(1, max_expand + 1):
        kappa, S_reg, lam_min, lam_max = probe(alpha_hi)
        if kappa <= target_kappa:
            break
        alpha_lo = alpha_hi
        alpha_hi *= 2.0
    else:
        raise RuntimeError(
            f"could not reach kappa <= {target_kappa} even at alpha={alpha_hi:.4g} "
            f"after {max_expand} doublings -- try a larger target_kappa."
        )

    # Bisect down to the smallest alpha in [alpha_lo, alpha_hi] that still
    # clears the target -- always keep the returned answer on the passing
    # side, never the failing one.
    n_bisect = 0
    for n_bisect in range(1, max_bisect + 1):
        mid = 0.5 * (alpha_lo + alpha_hi)
        kappa, S_reg, lam_min, lam_max = probe(mid)
        if kappa <= target_kappa:
            alpha_hi = mid
        else:
            alpha_lo = mid
        if alpha_hi - alpha_lo < bisect_tol:
            break

    kappa, S_reg, lam_min, lam_max = probe(alpha_hi)
    return alpha_hi, S_reg, {
        "kappa": kappa, "lambda_min": lam_min, "lambda_max": lam_max,
        "n_expand_steps": n_expand, "n_bisect_steps": n_bisect,
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Regularize a covariance/intercept matrix (or a "
                    "[target]+features submatrix of it) to be PD with a "
                    "bounded condition number, via ridge/floor/higham.",
    )
    parser.add_argument("--sigma", required=True, help="Path to the labelled matrix (e.g. official_intercept_matrix.csv)")
    parser.add_argument("--target", default=None, help="Trait to put first (row/col 0); requires --features. Omit to use the full matrix.")
    parser.add_argument("--features", nargs="*", default=None, help="Feature traits (order preserved). Omit with --target to use the full matrix.")
    parser.add_argument("--kappa", type=float, default=100.0, help="Target condition number ceiling (default: 100)")
    parser.add_argument("--method", choices=["ridge", "floor", "higham"], default="ridge")
    parser.add_argument("--out", default=None, help="Optional path to write the regularized matrix as labelled CSV")
    args = parser.parse_args()

    S_full, labels = load_matrix(args.sigma)

    if args.target is not None:
        if not args.features:
            raise SystemExit("--target requires --features")
        order = [args.target] + list(args.features)
        missing = [o for o in order if o not in labels]
        if missing:
            raise SystemExit(f"not found in {args.sigma}: {', '.join(missing)}")
        idx = [labels.index(o) for o in order]
        S = S_full[np.ix_(idx, idx)]
        out_labels = order
    else:
        S = S_full
        out_labels = labels
    S = 0.5 * (S + S.T)

    lam_min0, lam_max0 = np.linalg.eigvalsh(S)[0], np.linalg.eigvalsh(S)[-1]
    kappa0 = lam_max0 / lam_min0 if lam_min0 > 0 else float("inf")
    print(f"input:  K={S.shape[0]}, lambda_min={lam_min0:.6g}, lambda_max={lam_max0:.6g}, "
          f"kappa={kappa0:.6g}{' (not PD)' if lam_min0 <= 0 else ''}")

    alpha, S_reg, info = regularize_to_condition_number(
        S, target_kappa=args.kappa, method=args.method,
    )
    print(f"output: alpha={alpha:.6g} ({args.method}), lambda_min={info['lambda_min']:.6g}, "
          f"lambda_max={info['lambda_max']:.6g}, kappa={info['kappa']:.6g} "
          f"[{info['n_expand_steps']} expand + {info['n_bisect_steps']} bisect steps]")

    if args.out:
        pd.DataFrame(S_reg, index=out_labels, columns=out_labels).to_csv(args.out)
        print(f"wrote regularized matrix to {args.out}")


if __name__ == "__main__":
    _cli()
