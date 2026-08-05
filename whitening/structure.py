"""Structural prerequisites: alignment, symmetry, missing entries, diagonal.

These run before anything spectral.  A permutation mismatch or a triangle-only
matrix produces something that is still square, still positive definite, and
still whitens *something* — just not your data — so none of the numeric checks
downstream would catch it.
"""

from __future__ import annotations

import numpy as np

from .checks import Check


# ---------------------------------------------------------------------------
# 1. Column alignment
# ---------------------------------------------------------------------------

def check_alignment(
    sigma_labels: list[str],
    z_labels: list[str] | None,
    policy: str = "error",
) -> tuple[Check, list[int] | None]:
    """Verify Sigma's trait ordering against Z's columns, without ever
    shrinking Sigma.

    Whitening should be built from and applied to the *full* Sigma you loaded
    (all K traits) unless you explicitly restrict it via ``sigma.columns`` —
    a partial Z (fewer traits than Sigma) is common and is not, by itself, a
    reason to throw away the traits Z doesn't happen to cover.

    So this returns at most a **reorder** index (same trait set, different
    order — lossless, always safe to apply) and never a **restriction** index.
    A partial overlap is reported (fail or warn, per ``policy``) but Sigma
    passes through unchanged; nothing downstream currently multiplies ``W``
    against Z's actual values, so there is nothing that requires the two to
    share a trait set.

    ``policy``
        ``error`` – a differing label set fails.
        ``warn``  – a differing label set warns but Sigma still runs whole.
    """
    if z_labels is None:
        return Check(
            "1. column alignment", "skip",
            "No z-score matrix supplied — Sigma's trait ordering was not verified "
            "against any data. Set z.path before trusting a whitened result.",
            {"n_sigma_traits": len(sigma_labels)},
        ), None

    if sigma_labels == z_labels:
        return Check(
            "1. column alignment", "ok",
            f"Sigma and Z agree on all {len(sigma_labels)} trait labels, in order.",
            {"n_traits": len(sigma_labels)},
        ), None

    shared = [lab for lab in sigma_labels if lab in set(z_labels)]
    only_sigma = [lab for lab in sigma_labels if lab not in set(z_labels)]
    only_z = [lab for lab in z_labels if lab not in set(sigma_labels)]
    same_set = not only_sigma and not only_z

    details = {
        "n_sigma_traits": len(sigma_labels),
        "n_z_traits": len(z_labels),
        "n_shared": len(shared),
        "only_in_sigma": only_sigma[:25],
        "only_in_z": only_z[:25],
        "same_set_different_order": same_set,
    }

    if same_set:
        # Lossless: same K traits, just permuted. Reorder Sigma to Z's ordering
        # rather than the other way round, so a caller reading z_labels off
        # disk doesn't also need to re-derive a permutation of its own rows.
        s_idx = [sigma_labels.index(lab) for lab in z_labels]
        return Check(
            "1. column alignment", "warn",
            f"Same {len(shared)} traits in both, but in a different order. "
            "Reordered Sigma to Z's ordering — whitening without this would "
            "have been silently wrong.",
            details,
        ), s_idx

    status = "warn" if policy == "warn" else "fail"
    msg = (
        f"Label sets differ: {len(only_sigma)} trait(s) only in Sigma, "
        f"{len(only_z)} only in Z, {len(shared)} shared. "
    )
    if status == "fail":
        msg += "Set alignment.on_mismatch: warn to proceed anyway."
    else:
        msg += (
            "Proceeding with the FULL Sigma unchanged — the traits Z doesn't "
            "cover are simply not cross-checked against real data by checks "
            "1 or 7b; they still go through every other check (spectrum, "
            "PD-ness, calibration, ...) at full scope."
        )
    return Check("1. column alignment", status, msg, details), None


# ---------------------------------------------------------------------------
# 2. Symmetry
# ---------------------------------------------------------------------------

def check_symmetry(S: np.ndarray, tol: float = 1e-10) -> tuple[Check, np.ndarray]:
    """Report the asymmetry, then return the symmetrised matrix.

    ``eigh`` reads one triangle and says nothing, so an asymmetric input is
    diagonalised as if the other triangle did not exist.
    """
    A = np.abs(S - S.T)
    max_asym = float(np.nanmax(A))
    scale = float(np.nanmax(np.abs(S))) or 1.0
    rel = max_asym / scale
    i, j = np.unravel_index(np.nanargmax(A), A.shape)

    Ssym = 0.5 * (S + S.T)
    details = {
        "max_abs_asymmetry": max_asym,
        "relative_asymmetry": rel,
        "argmax_ij": [int(i), int(j)],
        "sigma_ij": float(S[i, j]),
        "sigma_ji": float(S[j, i]),
    }

    if max_asym == 0.0:
        return Check("2. symmetry", "ok", "Exactly symmetric.", details), Ssym
    if rel < tol:
        return Check(
            "2. symmetry", "ok",
            f"Symmetric to {max_asym:.2e} (round-off). Symmetrised anyway.",
            details,
        ), Ssym
    status = "warn" if rel < 1e-3 else "fail"
    return Check(
        "2. symmetry", status,
        f"Asymmetric by up to {max_asym:.3g} (relative {rel:.2e}) at "
        f"({i}, {j}): {S[i, j]:.4g} vs {S[j, i]:.4g}. Symmetrised as "
        "0.5*(S + S.T)" + ("." if status == "warn" else
        " — but a discrepancy this large is a construction bug, not round-off."),
        details,
    ), Ssym


# ---------------------------------------------------------------------------
# 3. Missing / defaulted entries
# ---------------------------------------------------------------------------

def check_missing(
    S: np.ndarray,
    labels: list[str],
    zero_is_default: bool = True,
    warn_frac: float = 0.01,
) -> tuple[Check, np.ndarray]:
    """Count NaNs and exact-zero off-diagonals.

    Exact zeros matter because a pairwise estimator that falls back to 0 when a
    pair has too few qualifying SNPs writes a *modelling choice* into a slot
    that reads like a measurement.  Enough of them distorts the spectrum.
    """
    K = S.shape[0]
    off = ~np.eye(K, dtype=bool)

    nan_mask = np.isnan(S)
    n_nan = int(nan_mask.sum())
    n_nan_off = int((nan_mask & off).sum())
    nan_rows = [labels[i] for i in np.where(nan_mask.any(axis=1))[0][:25]]

    zero_mask = (S == 0.0) & off
    n_zero = int(zero_mask.sum())
    zero_frac = n_zero / max(int(off.sum()), 1)
    per_trait_zeros = zero_mask.sum(axis=1)
    worst_traits = [
        {"trait": labels[i], "n_defaulted": int(per_trait_zeros[i])}
        for i in np.argsort(-per_trait_zeros)[:10]
        if per_trait_zeros[i] > 0
    ]

    S_out = S.copy()
    if n_nan:
        # Imputing 0 here is itself the choice the check is warning about; do it
        # so the spectrum is computable, and make the count impossible to miss.
        S_out[nan_mask] = 0.0
        np.fill_diagonal(S_out, np.where(np.isnan(np.diag(S)), 1.0, np.diag(S_out)))

    details = {
        "n_nan": n_nan,
        "n_nan_offdiag": n_nan_off,
        "traits_with_nan": nan_rows,
        "n_exact_zero_offdiag": n_zero,
        "frac_exact_zero_offdiag": zero_frac,
        "n_offdiag": int(off.sum()),
        "traits_most_defaulted": worst_traits,
        "zero_treated_as_default": zero_is_default,
    }

    if n_nan:
        return Check(
            "3. missing / defaulted entries", "fail",
            f"{n_nan} NaN entries ({n_nan_off} off-diagonal) across "
            f"{len(nan_rows)} trait(s). Imputed 0.0 so the spectrum is "
            "computable, but decide explicitly: drop those traits, or "
            "estimate the entries.",
            details,
        ), S_out

    if n_zero and zero_is_default:
        status = "warn" if zero_frac < warn_frac else "fail"
        return Check(
            "3. missing / defaulted entries", status,
            f"{n_zero} off-diagonal entries are exactly 0.0 "
            f"({zero_frac:.2%} of {int(off.sum())}). Exact zeros in a pairwise "
            "estimate are the too-few-qualifying-SNPs default, not a measured "
            "independence — they are imputed values in a matrix you are about "
            "to invert.",
            details,
        ), S_out

    return Check(
        "3. missing / defaulted entries", "ok",
        "No NaNs and no exact-zero off-diagonals.", details,
    ), S_out


# ---------------------------------------------------------------------------
# 4. Diagonal
# ---------------------------------------------------------------------------

def check_diagonal(S: np.ndarray, labels: list[str], tol: float = 1e-6) -> Check:
    """Is this a correlation matrix (diag == 1) or raw intercepts (1 + a_k)?

    When the diagonal is exactly 1, ZCA and ZCA-cor coincide and the transform
    choice is moot.  When it is not, V != I and you must decide explicitly
    whether to rescale — the decision changes what "whitened" means.
    """
    d = np.diag(S).astype(float)
    dev = np.abs(d - 1.0)
    max_dev = float(np.nanmax(dev))
    order = np.argsort(-dev)
    extremes = [{"trait": labels[i], "diag": float(d[i])} for i in order[:10]]

    details = {
        "diag_min": float(np.nanmin(d)),
        "diag_max": float(np.nanmax(d)),
        "diag_mean": float(np.nanmean(d)),
        "max_abs_deviation_from_one": max_dev,
        "n_below_one": int((d < 1.0 - tol).sum()),
        "n_nonpositive": int((d <= 0).sum()),
        "is_correlation_form": bool(max_dev <= tol),
        "furthest_from_one": extremes,
    }

    if (d <= 0).any():
        bad = [labels[i] for i in np.where(d <= 0)[0][:10]]
        return Check(
            "4. diagonal", "fail",
            f"{int((d <= 0).sum())} trait(s) have a non-positive variance on the "
            f"diagonal ({', '.join(bad)}). No valid covariance matrix has this; "
            "the estimator failed for those traits.",
            details,
        )

    if max_dev <= tol:
        return Check(
            "4. diagonal", "ok",
            "Diagonal is 1 to machine precision — correlation form, so "
            "ZCA == ZCA-cor and the transform choice is moot.",
            details,
        )

    below = int((d < 1.0 - tol).sum())
    msg = (
        f"Diagonal is not 1 (range {d.min():.4g} to {d.max():.4g}, max deviation "
        f"{max_dev:.4g}) — these are raw intercepts 1 + a_k, not correlations. "
        "V != I, so ZCA and ZCA-cor differ: decide explicitly via "
        "transform / standardize in the config."
    )
    if below:
        msg += (
            f" {below} trait(s) sit below 1, which an intercept 1 + a_k with "
            "a_k >= 0 cannot do — that is estimation noise on the diagonal."
        )
    return Check("4. diagonal", "warn", msg, details)


def to_correlation(S: np.ndarray) -> np.ndarray:
    """Rescale a covariance to correlation form (unit diagonal)."""
    d = np.sqrt(np.diag(S))
    if np.any(d <= 0):
        raise ValueError("cannot rescale to correlation form: non-positive diagonal")
    return S / np.outer(d, d)
