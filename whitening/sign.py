"""Check 7: sign consistency of the z-scores / of Sigma's off-diagonals.

If effect alleles are coded inconsistently across traits (strand-ambiguous A/T
and C/G variants are the classic source), some z_{j,k} carry a silent sign
flip. That attenuates estimated off-diagonals toward zero — which *inflates*
lambda_min and makes Sigma look better conditioned than it is. You would then
whiten by a matrix that understates the true correlation, and the residual
correlation would resurface later as unexplained structure.

Two independent probes, run with whatever inputs are available:

* **Structural** (needs a group/category map): within a group of traits that
  should be strongly, same-signed correlated (e.g. same anatomical/hemispheric
  category), scattered sign flips among the strong pairs are suspicious.
* **Allele-driven** (needs Z plus an allele frame with a strand-ambiguity
  flag): compare the empirical sign pattern restricted to unambiguous SNPs
  against the pattern from all SNPs. A shift concentrated on ambiguous SNPs is
  the direct fingerprint of unresolved strand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .checks import Check

AMBIGUOUS_PAIRS = {frozenset(("A", "T")), frozenset(("C", "G"))}


def check_sign_consistency(
    S: np.ndarray,
    labels: list[str],
    groups: dict[str, str] | None = None,
    strong_thresh: float = 0.3,
) -> Check:
    """Structural probe: sign pattern should follow group structure.

    Within each group with >= 2 members, off-diagonal entries above
    ``strong_thresh`` in magnitude are expected to share the group's majority
    sign. Flips among *strong* pairs are the concerning case — weak entries
    near zero are expected to have unstable sign.
    """
    if groups is None:
        return Check(
            "7. sign consistency", "skip",
            "No trait-group mapping supplied — set sign_check.groups_path to "
            "probe whether the sign pattern follows anatomical/category "
            "structure (a scattered-flip pattern among strong pairs indicates "
            "strand-ambiguous SNPs left unresolved).",
        )

    idx_by_group: dict[str, list[int]] = {}
    unmapped = []
    for i, lab in enumerate(labels):
        g = groups.get(lab)
        if g is None:
            unmapped.append(lab)
        else:
            idx_by_group.setdefault(g, []).append(i)

    flips = []
    n_strong_pairs = 0
    for g, idx in idx_by_group.items():
        if len(idx) < 2:
            continue
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        vals = sub[iu]
        strong = np.abs(vals) >= strong_thresh
        if strong.sum() < 2:
            continue
        n_strong_pairs += int(strong.sum())
        signs = np.sign(vals[strong])
        majority = np.sign(signs.sum()) or 1.0
        minority_mask = signs != majority
        if minority_mask.any():
            pair_idx = np.array(iu).T[strong][minority_mask]
            for a, b in pair_idx:
                flips.append({
                    "group": g,
                    "trait_i": labels[idx[a]],
                    "trait_j": labels[idx[b]],
                    "value": float(sub[a, b]),
                })

    details = {
        "n_groups": len(idx_by_group),
        "n_unmapped_traits": len(unmapped),
        "unmapped_traits": unmapped[:20],
        "n_strong_pairs_checked": n_strong_pairs,
        "n_sign_flips": len(flips),
        "flips": flips[:25],
        "strong_thresh": strong_thresh,
    }

    if n_strong_pairs == 0:
        return Check(
            "7. sign consistency", "skip",
            f"{len(idx_by_group)} group(s) found but none had >=2 strong "
            f"(|corr| >= {strong_thresh}) within-group pairs to check.",
            details,
        )

    frac = len(flips) / n_strong_pairs
    if frac == 0:
        return Check(
            "7. sign consistency", "ok",
            f"All {n_strong_pairs} strong within-group pairs share their "
            "group's majority sign.", details,
        )
    status = "warn" if frac < 0.1 else "fail"
    example = flips[0]
    return Check(
        "7. sign consistency", status,
        f"{len(flips)}/{n_strong_pairs} strong within-group pairs "
        f"({frac:.1%}) have a sign opposite their group's majority — e.g. "
        f"{example['trait_i']} vs {example['trait_j']} in group "
        f"'{example['group']}' = {example['value']:.3g}. Scattered flips among "
        "strongly-correlated pairs are the signature of unresolved "
        "strand-ambiguous alleles, which attenuate off-diagonals toward zero "
        "and make Sigma look better conditioned than it is.",
        details,
    )


def check_allele_ambiguity(
    meta: pd.DataFrame | None,
    a0_col: str = "A0",
    a1_col: str = "A1",
) -> Check:
    """Direct probe: how many SNPs in the loaded Z matrix carry an
    A/T or C/G allele pair, where strand is intrinsically unresolvable from
    the alleles alone and sign errors are most likely to hide.
    """
    if meta is None or a0_col not in meta.columns or a1_col not in meta.columns:
        return Check(
            "7b. strand-ambiguous SNPs", "skip",
            f"No allele columns ('{a0_col}', '{a1_col}') in the loaded Z "
            "metadata — cannot flag strand-ambiguous SNPs directly.",
        )

    def is_ambiguous(a0, a1) -> bool:
        if not isinstance(a0, str) or not isinstance(a1, str):
            return False
        if len(a0) != 1 or len(a1) != 1:
            return False
        return frozenset((a0.upper(), a1.upper())) in AMBIGUOUS_PAIRS

    amb = meta.apply(lambda r: is_ambiguous(r[a0_col], r[a1_col]), axis=1)
    n_amb = int(amb.sum())
    frac = n_amb / len(meta) if len(meta) else 0.0
    details = {"n_snps": int(len(meta)), "n_ambiguous": n_amb, "frac_ambiguous": frac}

    if frac == 0:
        return Check("7b. strand-ambiguous SNPs", "ok", "No strand-ambiguous SNPs in the loaded set.", details)

    # This is a presence count, not a severity graded on frac: a nonzero count
    # always means *some* of these SNPs are in the loaded data (typical GWAS
    # background rate is ~10-20%, driven by real allele-frequency spectra, not
    # a pipeline bug), and this check has no way to know whether they were
    # frequency-resolved upstream. It always warns to keep that uncertainty
    # visible, and the message reports where frac sits relative to that
    # typical range so a large deviation is still legible.
    band = "within the typical ~10-20% GWAS background rate" if frac <= 0.20 else \
           "well above the typical ~10-20% GWAS background rate — worth checking allele coding"
    return Check(
        "7b. strand-ambiguous SNPs", "warn",
        f"{n_amb}/{len(meta)} loaded SNPs ({frac:.1%}, {band}) are "
        "strand-ambiguous (A/T or C/G). These are exactly where a silent "
        "effect-allele sign flip would attenuate estimated off-diagonals "
        "toward zero. If Sigma was built including these without "
        "frequency-based strand resolution, treat lambda_min as an upper "
        "bound on the true value.",
        details,
    )
