"""Check 8: calibration — the acid test, and the sweep that turns
regularisation into a defensible tuned parameter instead of an arbitrary fix.

Whitening is supposed to do one thing: turn null z-scores (Cov = Sigma) into
w = W z with Cov(w) = I. Everything upstream (alignment, symmetry, the
diagonal, PD-ness) is a precondition for this test, not a substitute for it.

Three things get checked, in increasing order of how easy they are to miss:

1. Cov(w) ~= I               — off-diagonals near 0, diagonal near 1.
2. E[|w|^2] ~= K              — deviation *above* K is the Jensen-gap inflation
                                 from inverting a noisy Sigma-hat (E[Sigma^{-1}]
                                 has more mass than Sigma^{-1} in expectation).
3. tail(|w|^2) vs chi^2_K     — mean calibration can look fine while the
                                 1e-8 quantile is off by orders of magnitude,
                                 and the tail is where decisions get made.

Null SNPs are simulated as z ~ N(0, Sigma) (matching the source report's
Sec 8.6) rather than pulled from real "should-be-null" SNPs, because that
needs an LD-pruned real null set this module has no path to; real SNPs are
expected to show residual off-diagonal structure (genuine genetic covariance)
that would contaminate this test, which is exactly why simulated null is the
clean reference here.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .checks import Check
from .spectrum import build_whitener, regularize


def simulate_null(S: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Draw z ~ N(0, S), n samples x K traits."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(S.shape[0]), S, size=n, method="eigh")


def calibration_metrics(w: np.ndarray) -> dict:
    K = w.shape[1]
    cov = np.cov(w, rowvar=False)
    off = cov[~np.eye(K, dtype=bool)]
    quad = np.einsum("ij,ij->i", w, w)  # |w|^2 per sample

    quantiles = [1e-1, 1e-2, 1e-3, 1e-4]
    n = len(quad)
    tail = {}
    for q in quantiles:
        # Empirical q-quantile of the *upper* tail vs the chi^2_K reference.
        if n * q < 5:
            continue
        emp = float(np.quantile(quad, 1 - q))
        ref = float(stats.chi2.ppf(1 - q, df=K))
        tail[q] = {"empirical": emp, "chi2_reference": ref, "ratio": emp / ref if ref else np.nan}

    return {
        "K": K,
        "n": n,
        "diag_mean": float(np.mean(np.diag(cov))),
        "diag_std": float(np.std(np.diag(cov))),
        "offdiag_mean_abs": float(np.mean(np.abs(off))),
        "offdiag_max_abs": float(np.max(np.abs(off))),
        "mean_quad_form": float(np.mean(quad)),
        "expected_quad_form": float(K),
        "quad_form_inflation": float(np.mean(quad) / K),
        "tail": tail,
    }


def check_calibration(
    S: np.ndarray,
    transform: str = "zca",
    n_null: int = 20000,
    seed: int = 0,
    offdiag_tol: float = 0.05,
    inflation_tol: float = 1.15,
    tail_ratio_tol: float = 2.0,
) -> tuple[Check, dict]:
    z = simulate_null(S, n_null, seed=seed)
    W = build_whitener(S, transform=transform)
    w = whiten_rows(z, W)
    m = calibration_metrics(w)

    bad_tail = {q: v for q, v in m["tail"].items() if v["ratio"] > tail_ratio_tol or v["ratio"] < 1 / tail_ratio_tol}

    issues = []
    if m["offdiag_mean_abs"] > offdiag_tol:
        issues.append(f"mean |off-diagonal| = {m['offdiag_mean_abs']:.3g} > {offdiag_tol}")
    if m["quad_form_inflation"] > inflation_tol:
        issues.append(
            f"E[|w|^2]/K = {m['quad_form_inflation']:.3g} > {inflation_tol} "
            "(Jensen inflation from inverting a noisy Sigma-hat)"
        )
    if bad_tail:
        worst_q = min(bad_tail, key=lambda q: -abs(np.log(bad_tail[q]["ratio"])))
        issues.append(
            f"tail miscalibrated at q={worst_q:g}: empirical/chi2 ratio = "
            f"{bad_tail[worst_q]['ratio']:.3g}"
        )

    status = "ok" if not issues else ("warn" if len(issues) == 1 else "fail")
    msg = (
        f"[{transform}] E[|w|^2]/K = {m['quad_form_inflation']:.3f} "
        f"(target 1.0), mean|off-diag Cov(w)| = {m['offdiag_mean_abs']:.3g} "
        f"(target 0), diag Cov(w) = {m['diag_mean']:.3f}+/-{m['diag_std']:.3f}."
    )
    if issues:
        msg += " Issues: " + "; ".join(issues) + "."
    else:
        msg += " Mean and tail calibration both pass on simulated null."
    msg += (
        " (On real SNPs expect residual off-diagonal structure — that is "
        "genetic covariance, which whitening by the intercept is supposed to "
        "leave behind; this test uses simulated null specifically to avoid "
        "that contamination.)"
    )

    return Check("8. null calibration", status, msg, m), m


def whiten_rows(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    return Z @ W.T


# ---------------------------------------------------------------------------
# Regularisation sweep — turns an arbitrary numerical fix into a tuned,
# reported parameter, per the source report's closing recommendation.
# ---------------------------------------------------------------------------

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

    ``sample_from`` lets the caller supply a already-regularised, guaranteed-PD
    matrix to draw the shared null z's from when ``S`` itself is not PD (as at
    the smallest end of a sweep that intentionally includes alpha=0). Defaults
    to ``S``.
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
                "alpha": alpha,
                "feasible": True,
                "lambda_min": lam_min,
                "condition_number": float(np.linalg.eigvalsh(S_reg)[-1] / lam_min),
                "quad_form_inflation": m["quad_form_inflation"],
                "offdiag_mean_abs": m["offdiag_mean_abs"],
                "tail_ok": tail_ok,
                "tail": m["tail"],
            })
        except np.linalg.LinAlgError as e:
            rows.append({"alpha": alpha, "feasible": False, "error": str(e)})

    feasible = [r for r in rows if r.get("feasible") and r.get("tail_ok")]
    recommended = min(feasible, key=lambda r: r["alpha"])["alpha"] if feasible else None

    return {
        "method": method,
        "transform": transform,
        "alphas": alphas,
        "results": rows,
        "recommended_alpha": recommended,
        "note": (
            "Smallest alpha for which both the covariance and the chi^2 tail "
            "of the whitened simulated-null draws calibrate. Report this "
            "sweep alongside the choice — that is what makes alpha a tuned "
            "parameter rather than an arbitrary fix."
            if recommended is not None else
            "No alpha in the sweep achieved tail calibration — widen the "
            "range in whitening.sweep.alphas."
        ),
    }
