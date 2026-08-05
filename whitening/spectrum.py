"""Spectral diagnostics, regularisation, and the whitening transforms.

The framing that drives all of this: kappa is *not* the requirement.  If Sigma
were known exactly, kappa would be irrelevant — the lambda_i in the numerator
of (v_i' z)^2 cancels the lambda_i^{-1} exactly, and a correctly specified
whitening of a matrix with kappa = 1e6 is perfectly calibrated.  What matters
is ||E||_2 / lambda_min: the size of the estimation error relative to the
smallest thing you are about to divide by.  kappa is a proxy for that ratio,
not the ratio itself.

So the spectrum checks below report kappa, but they *decide* on lambda_min,
the shape of the bottom of the spectrum, and (in probe.py) a direct estimate
of ||E||_2.
"""

from __future__ import annotations

import numpy as np

from .checks import Check


# ---------------------------------------------------------------------------
# 5. Full spectrum
# ---------------------------------------------------------------------------

def check_spectrum(S: np.ndarray, gap_ratio: float = 5.0) -> tuple[Check, dict]:
    """Eigenvalues, effective rank, and the bulk-vs-smooth-decay question.

    A clean bulk with a few small outliers means truncation has a natural
    cutoff.  A smooth decay to zero means it does not, and any threshold you
    pick is arbitrary — ridge is the more honest fix there.
    """
    lam, V = np.linalg.eigh(S)          # ascending
    K = len(lam)
    trace = float(lam.sum())
    lam_pos = np.clip(lam, 0.0, None)

    # Participation ratio. Computed on the clipped spectrum so a negative
    # eigenvalue does not silently shrink the numerator.
    eff_rank = float(lam_pos.sum() ** 2 / np.sum(lam_pos ** 2)) if np.any(lam_pos > 0) else 0.0

    n_bottom = max(1, int(np.ceil(K / 10)))
    bottom_trace_frac = float(lam_pos[:n_bottom].sum() / lam_pos.sum()) if lam_pos.sum() > 0 else np.nan

    kappa = float(lam[-1] / lam[0]) if lam[0] != 0 else np.inf

    # Largest multiplicative gap in the bottom half of the spectrum.
    half = max(2, K // 2)
    pos = lam[:half]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(pos[:-1] > 0, pos[1:] / pos[:-1], np.inf)
    gap_at = int(np.nanargmax(ratios)) + 1 if len(ratios) else 0
    gap_size = float(np.nanmax(ratios)) if len(ratios) else 1.0
    clean_bulk = bool(np.isfinite(gap_size) and gap_size >= gap_ratio and gap_at <= K // 4)

    spec = {
        "eigenvalues": lam,
        "eigenvectors": V,
        "K": K,
        "lambda_min": float(lam[0]),
        "lambda_max": float(lam[-1]),
        "condition_number": kappa,
        "trace": trace,
        "effective_rank": eff_rank,
        "n_bottom_decile": n_bottom,
        "trace_frac_bottom_decile": bottom_trace_frac,
        "n_nonpositive": int((lam <= 0).sum()),
        "gap_index": gap_at,
        "gap_ratio": gap_size,
        "clean_bulk": clean_bulk,
        "smallest_10": lam[:10],
        "largest_10": lam[-10:][::-1],
    }

    details = {k: v for k, v in spec.items() if k not in {"eigenvectors"}}

    shape = (
        f"clean bulk with {gap_at} eigenvalue(s) below a {gap_size:.1f}x gap — "
        "truncation has a natural cutoff there"
        if clean_bulk else
        f"smooth decay (largest bottom-half gap only {gap_size:.2f}x) — no "
        "natural cutoff, so truncation would be arbitrary and ridge is the "
        "more honest fix"
    )
    msg = (
        f"K={K}, lambda in [{lam[0]:.4g}, {lam[-1]:.4g}], kappa={kappa:.3g}, "
        f"effective rank {eff_rank:.1f}/{K}, "
        f"{bottom_trace_frac:.2%} of the trace in the bottom decile. Spectrum "
        f"shape: {shape}."
    )
    status = "ok" if lam[0] > 0 else "warn"   # positive-definiteness is check 6
    return Check("5. spectrum", status, msg, details), spec


# ---------------------------------------------------------------------------
# 6. Positive definiteness
# ---------------------------------------------------------------------------

def check_positive_definite(spec: dict, eps_rel: float = 1e-8) -> Check:
    """A negative eigenvalue is evidence about ||E||_2, not just a numerical nit.

    The estimand is positive definite.  If the estimate is not, then the
    estimation error exceeded the true smallest eigenvalue in that direction —
    ||E||_2 > lambda_min^true.  That is a statement about the data, and it is
    the same statement check 7 (the two-estimate probe) makes quantitatively.
    """
    lam = spec["eigenvalues"]
    lam_min, lam_max = spec["lambda_min"], spec["lambda_max"]
    eps = eps_rel * lam_max
    n_neg = int((lam < 0).sum())
    n_tiny = int(((lam >= 0) & (lam < eps)).sum())

    details = {
        "lambda_min": lam_min,
        "n_negative": n_neg,
        "n_below_eps": n_tiny,
        "eps": float(eps),
        "negative_mass": float(-lam[lam < 0].sum()) if n_neg else 0.0,
        "distance_to_psd_spectral": float(max(0.0, -lam_min)),
    }

    if n_neg:
        return Check(
            "6. positive definiteness", "fail",
            f"Not positive semi-definite: {n_neg} negative eigenvalue(s), "
            f"lambda_min = {lam_min:.4g} (negative mass {details['negative_mass']:.4g}). "
            "This is not a rounding artifact to clip past — the estimand is PD, "
            "so a negative eigenvalue says the estimation error exceeded the "
            "true eigenvalue in that direction. Regularise (ridge / eigenvalue "
            "floor / Higham nearest-PD) and tune the strength by null "
            "calibration, not by eye.",
            details,
        )
    if lam_min < eps:
        return Check(
            "6. positive definiteness", "warn",
            f"PSD but numerically singular: lambda_min = {lam_min:.4g} < "
            f"{eps:.3g} (1e-8 * lambda_max). Dividing by this is dividing by "
            "noise; regularise.",
            details,
        )
    return Check(
        "6. positive definiteness", "ok",
        f"Positive definite, lambda_min = {lam_min:.4g}.", details,
    )


# ---------------------------------------------------------------------------
# Regularisers
# ---------------------------------------------------------------------------

def regularize(
    S: np.ndarray,
    method: str = "ridge",
    alpha: float = 0.0,
    renormalize: bool | None = None,
) -> np.ndarray:
    """Return a regularised copy of ``S``.

    ``ridge``
        ``S + alpha * I``. Uniform, preserves diagonal homogeneity, no cutoff to
        justify. With ``renormalize`` the result is divided by ``1 + alpha`` so a
        correlation matrix stays unit-diagonal.
    ``floor``
        ``lambda -> max(lambda, alpha * lambda_max)``. Targeted at the offending
        directions, but breaks diagonal homogeneity.
    ``higham``
        Nearest positive-definite matrix by alternating projections (Higham
        2002), holding the diagonal fixed. ``alpha`` becomes the eigenvalue
        floor of the projection step.
    ``none``
        Passthrough.
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
    """Higham's nearest PD matrix with the diagonal held fixed.

    Alternating projections between the PSD cone (eigenvalue floor at ``eps``)
    and the set of matrices with ``S``'s diagonal.  With a unit diagonal this is
    the classic nearest-correlation-matrix algorithm.
    """
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
    if lam[0] <= 0:  # diagonal constraint can push it back; nudge minimally
        Y = Y + (eps - lam[0] + 1e-12) * np.eye(len(Y))
    return 0.5 * (Y + Y.T)


# ---------------------------------------------------------------------------
# Whitening transforms
# ---------------------------------------------------------------------------

def build_whitener(S: np.ndarray, transform: str = "zca") -> np.ndarray:
    """Return ``W`` with ``Cov(W z) = I`` when ``Cov(z) = S``.

    ``zca``      W = S^{-1/2}                (symmetric, minimal rotation)
    ``zca-cor``  W = R^{-1/2} V^{-1/2}       (standardise first, then whiten)
    ``pca``      W = Lambda^{-1/2} U'        (rotates into the eigenbasis)
    ``cholesky`` W = L^{-1}, S = L L'        (triangular, order-dependent)

    All four give an identity covariance under a correct S; they differ in
    which basis the whitened coordinates live in — which matters only once S is
    misspecified, and matters most in the small-lambda directions.
    """
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
            f"cannot whiten: lambda_min = {lam[0]:.4g} <= 0. Regularise first "
            "(whitening.regularization in the config)."
        )


def whiten(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply the whitener row-wise: ``w_j = W z_j``."""
    return Z @ W.T
