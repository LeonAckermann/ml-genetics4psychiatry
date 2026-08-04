"""The two-estimate error probe.

This is the single most informative number in the whole checklist, and it
costs one line: if you have two estimates of Sigma, ``||Sigma1 - Sigma2||_2``
is a direct read on the estimation error you would otherwise have to guess at
from kappa.

    If ||D||_2 >~ lambda_min, the small-eigenvalue subspace is not identified
    by your data: the two estimates disagree by more than the quantity you'd
    be dividing by, so whichever one you pick, the bottom of the spectrum is
    an artifact of the estimator, not a property of the traits.

Caveat this module cannot check for you: this only works if both estimates
target the same estimand and differ by noise (e.g. two null-SNP-filter
variants, or LDSC on two SNP subsets). If one is a naive empirical z-score
correlation and the other is LDSC/JASS-filtered, the difference is dominated
by bias (the l_bar * Sigma_g contamination), not sampling error, and
``||D||_2`` overstates the noise. The config requires the caller to say which
case this is (``probe.same_estimand: true/false``) and the report reflects it.
"""

from __future__ import annotations

import numpy as np

from .checks import Check


def two_estimate_probe(
    S1: np.ndarray,
    S2: np.ndarray,
    lambda_min: float,
    labels: list[str],
    same_estimand: bool = True,
    n_bottom: int = 5,
) -> tuple[Check, dict]:
    if S1.shape != S2.shape:
        raise ValueError(f"shape mismatch: {S1.shape} vs {S2.shape}")

    D = S1 - S2
    spec_norm = float(np.linalg.norm(D, 2))
    max_abs = float(np.abs(D).max())
    i, j = np.unravel_index(np.argmax(np.abs(D)), D.shape)

    lam1 = np.linalg.eigvalsh(0.5 * (S1 + S1.T))
    lam2 = np.linalg.eigvalsh(0.5 * (S2 + S2.T))
    ratio = spec_norm / lambda_min if lambda_min > 0 else np.inf

    # Eigenvector alignment in the bottom-lambda subspace (Davis-Kahan symptom).
    _, V1 = np.linalg.eigh(0.5 * (S1 + S1.T))
    _, V2 = np.linalg.eigh(0.5 * (S2 + S2.T))
    n_bottom = min(n_bottom, V1.shape[1])
    alignments = [float(abs(np.dot(V1[:, k], V2[:, k]))) for k in range(n_bottom)]

    details = {
        "spectral_norm_diff": spec_norm,
        "max_abs_entrywise_diff": max_abs,
        "argmax_entry": {"i": labels[i], "j": labels[j], "S1": float(S1[i, j]), "S2": float(S2[i, j])},
        "lambda_min_1": float(lam1[0]),
        "lambda_min_2": float(lam2[0]),
        "lambda_min_used": lambda_min,
        "ratio_spectral_norm_to_lambda_min": ratio,
        "bottom_eigvec_alignment": alignments,
        "bottom_eigvec_alignment_labels": [f"v_{k}" for k in range(n_bottom)],
        "same_estimand": same_estimand,
    }

    low_align = [a for a in alignments if a < 0.5]
    caveat = (
        "" if same_estimand else
        " CAVEAT: probe.same_estimand is False — the two estimates differ by "
        "construction (e.g. naive z-correlation vs LDSC/JASS), so this "
        "difference is dominated by bias (l_bar*Sigma_g contamination), not "
        "sampling noise, and overstates the true estimation error. Treat this "
        "as an upper bound, not a noise estimate; rerun with two variants of "
        "the same estimator for a clean read."
    )

    if not np.isfinite(ratio):
        status = "fail"
    elif ratio >= 1.0:
        status = "fail"
    elif ratio >= 0.3:
        status = "warn"
    else:
        status = "ok"

    msg = (
        f"||Sigma1 - Sigma2||_2 = {spec_norm:.4g} (max entry diff {max_abs:.4g} "
        f"at [{labels[i]}, {labels[j]}]), lambda_min = {lambda_min:.4g} -> "
        f"ratio {ratio:.3g}. "
    )
    if status == "ok":
        msg += "Estimation error is well below lambda_min: the bottom of the spectrum looks identified."
    elif status == "warn":
        msg += "Estimation error is a non-trivial fraction of lambda_min: treat the bottom eigenvalues with caution and check the sensitivity of downstream results to regularisation."
    else:
        msg += (
            "Estimation error is comparable to or exceeds lambda_min: the "
            "small-eigenvalue subspace is NOT identified by the data — whichever "
            "estimate you whiten by, the bottom of the spectrum is an artifact "
            "of the estimator, not a property of the traits. Regularise."
        )
    if low_align:
        msg += (
            f" Additionally, {len(low_align)}/{n_bottom} bottom eigenvector(s) "
            f"have |v1'v2| < 0.5 (near-orthogonal) — the Davis-Kahan symptom, "
            "confirming those directions are noise, not signal."
        )
    msg += caveat

    return Check("0. two-estimate error probe", status, msg, details), {"D": D, "spec_norm": spec_norm}
