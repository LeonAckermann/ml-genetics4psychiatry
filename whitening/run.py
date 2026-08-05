"""Whitening pre-flight — orchestrates every check from the checklist and
writes a report.

Usage
-----
    python -m whitening.run --config whitening/configs/toy.yaml
    python -m whitening.run --config whitening/configs/full_intercept_matrix.yaml

The config points at Sigma (required), and optionally at a second Sigma
estimate, a Z-score matrix, and a trait-group map — each additional input
unlocks the checks that need it (see whitening/configs/full_intercept_matrix.yaml
for every option, and whitening/README.md for what each check does and why).

Exit code is 1 if any check fails, 2 if the config/inputs are broken outright,
0 otherwise — wire this into a batch script with ``&&`` to gate later stages
on a clean whitening report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from .calibration import check_calibration, sweep_regularization
from .checks import Check, worst
from .io import load_groups, load_matrix, load_z
from .probe import two_estimate_probe
from .report import write_report
from .sign import check_allele_ambiguity, check_sign_consistency
from .spectrum import check_positive_definite, check_spectrum, regularize
from .structure import check_alignment, check_diagonal, check_missing, check_symmetry, to_correlation


def run(cfg: dict) -> tuple[list[Check], dict, dict]:
    """Run every check the config's inputs support.

    Returns ``(checks, meta, artifacts)`` where ``artifacts`` carries the
    regularised Sigma, the spectrum, and calibration metrics for callers (e.g.
    a plotting step or a follow-on pipeline stage) that want the numbers, not
    just the verdicts.
    """
    checks: list[Check] = []
    artifacts: dict = {}

    sigma_cfg = cfg["sigma"]
    S1, labels = load_matrix(
        sigma_cfg["path"],
        labels_path=sigma_cfg.get("labels_path"),
        columns=sigma_cfg.get("columns"),
    )
    meta = {"sigma_path": sigma_cfg["path"], "n_traits": len(labels), "traits": labels}

    # ---- Z (optional) ------------------------------------------------------
    z_cfg = cfg.get("z")
    Z = z_labels = z_meta = z_stats = None
    if z_cfg and z_cfg.get("path"):
        Z, z_labels, z_meta, z_stats = load_z(
            z_cfg["path"],
            id_cols=z_cfg.get("id_cols", ["ID", "A0", "A1"]),
            columns=z_cfg.get("columns"),
            max_snps=z_cfg.get("max_snps", 200_000),
        )
        meta["z_path"] = z_cfg["path"]
        meta["z_stats"] = z_stats

    # ---- 1. alignment -------------------------------------------------------
    # Only ever reorders Sigma (lossless, same trait set) — never restricts it.
    # See structure.check_alignment's docstring for why a partial Z is not, by
    # itself, a reason to shrink the matrix everything else below whitens.
    align_check, reorder_idx = check_alignment(
        labels, z_labels, policy=cfg.get("alignment", {}).get("on_mismatch", "error"),
    )
    checks.append(align_check)
    if reorder_idx is not None:
        S1 = S1[np.ix_(reorder_idx, reorder_idx)]
        labels = [labels[i] for i in reorder_idx]
    elif align_check.status == "fail":
        return checks, meta, artifacts

    # ---- 2. symmetry ----------------------------------------------------
    sym_check, S1 = check_symmetry(S1, tol=cfg.get("symmetry", {}).get("tol", 1e-10))
    checks.append(sym_check)

    # ---- 3. missing / defaulted entries --------------------------------
    miss_check, S1 = check_missing(
        S1, labels,
        zero_is_default=cfg.get("missing", {}).get("zero_is_default", True),
        warn_frac=cfg.get("missing", {}).get("warn_frac", 0.01),
    )
    checks.append(miss_check)

    # ---- 4. diagonal ------------------------------------------------------
    diag_check = check_diagonal(S1, labels, tol=cfg.get("diagonal", {}).get("tol", 1e-6))
    checks.append(diag_check)

    transform_cfg = cfg.get("transform", {})
    want_correlation = transform_cfg.get("standardize", "auto")
    S_work = S1
    if want_correlation is True or (want_correlation == "auto" and diag_check.status == "warn"):
        S_work = to_correlation(S1)
        meta["rescaled_to_correlation_form"] = True

    # ---- 5. full spectrum -------------------------------------------------
    spec_check, spec = check_spectrum(S_work, gap_ratio=cfg.get("spectrum", {}).get("gap_ratio", 5.0))
    checks.append(spec_check)
    artifacts["spectrum"] = spec

    # ---- 6. positive definiteness -----------------------------------------
    pd_check = check_positive_definite(spec, eps_rel=cfg.get("spectrum", {}).get("eps_rel", 1e-8))
    checks.append(pd_check)

    # ---- 0. two-estimate error probe (optional second Sigma) --------------
    probe_cfg = cfg.get("probe", {})
    if probe_cfg.get("sigma2_path"):
        S2, labels2 = load_matrix(probe_cfg["sigma2_path"], columns=labels)
        if labels2 != labels:
            checks.append(Check(
                "0. two-estimate error probe", "fail",
                "Sigma2's trait labels do not match Sigma1's after column "
                "selection — cannot compare. Check probe.sigma2_path.",
            ))
        else:
            probe_check, probe_art = two_estimate_probe(
                S_work, S2, lambda_min=spec["lambda_min"], labels=labels,
                same_estimand=probe_cfg.get("same_estimand", True),
            )
            checks.append(probe_check)
            artifacts["probe"] = probe_art

    # ---- 7. sign consistency ------------------------------------------------
    sign_cfg = cfg.get("sign_check", {})
    groups = None
    if sign_cfg.get("groups_path"):
        groups = load_groups(
            sign_cfg["groups_path"],
            key_col=sign_cfg.get("key_col", "trait"),
            group_col=sign_cfg.get("group_col", "group"),
            key_sep=sign_cfg.get("key_sep", "_"),
        )
    checks.append(check_sign_consistency(S_work, labels, groups=groups,
                                          strong_thresh=sign_cfg.get("strong_thresh", 0.3)))
    if z_meta is not None:
        checks.append(check_allele_ambiguity(z_meta))

    # ---- regularisation for the calibration test ---------------------------
    reg_cfg = cfg.get("regularization", {})
    method = reg_cfg.get("method", "ridge")
    alpha = reg_cfg.get("alpha", 0.0)
    if pd_check.status != "ok" and alpha == 0.0 and method != "none":
        # Auto-floor with margin so simulate_null's Cholesky/eigh draw doesn't
        # land on a numerically-zero lambda_min; the sweep below is what
        # actually picks a defensible alpha, this just keeps check 8 runnable.
        target_lambda_min = 1e-2 * float(spec["lambda_max"])
        alpha = float(max(0.0, target_lambda_min - spec["lambda_min"]))
    S_reg = regularize(S_work, method=method, alpha=alpha) if alpha or method == "higham" else S_work
    artifacts["S_regularized"] = S_reg
    artifacts["S_work"] = S_work
    artifacts["labels"] = labels

    # ---- 8. null calibration -----------------------------------------------
    cal_cfg = cfg.get("calibration", {})
    if cal_cfg.get("run", True):
        cal_check, cal_metrics = check_calibration(
            S_reg,
            transform=transform_cfg.get("transform", "zca"),
            n_null=cal_cfg.get("n_null", 20000),
            seed=cal_cfg.get("seed", 0),
            offdiag_tol=cal_cfg.get("offdiag_tol", 0.05),
            inflation_tol=cal_cfg.get("inflation_tol", 1.15),
            tail_ratio_tol=cal_cfg.get("tail_ratio_tol", 2.0),
        )
        checks.append(cal_check)
        artifacts["calibration"] = cal_metrics

    # ---- regularisation sweep (optional) ------------------------------------
    sweep_cfg = cfg.get("sweep", {})
    if sweep_cfg.get("run", False):
        sweep = sweep_regularization(
            S_work, alphas=sweep_cfg.get("alphas", [0.0, 1e-3, 1e-2, 0.05, 0.1, 0.2]),
            method=sweep_cfg.get("method", method), transform=transform_cfg.get("transform", "zca"),
            n_null=sweep_cfg.get("n_null", cal_cfg.get("n_null", 20000)),
            seed=sweep_cfg.get("seed", 0),
            tail_ratio_tol=sweep_cfg.get("tail_ratio_tol", 2.0),
            sample_from=S_reg,
        )
        artifacts["sweep"] = sweep
        rec = sweep["recommended_alpha"]
        checks.append(Check(
            "sweep: regularization strength", "ok" if rec is not None else "warn",
            (f"Recommended alpha = {rec:g} ({sweep_cfg.get('method', method)}): "
             "smallest value in the sweep with full null-calibration."
             if rec is not None else sweep["note"]),
            {"recommended_alpha": rec, "alphas": sweep["alphas"]},
        ))

    return checks, meta, artifacts


def _make_plots(checks, artifacts, out_dir: Path) -> None:
    try:
        from . import plot
    except ImportError:
        checks.append(Check("plots", "skip", "matplotlib not available — skipped plots."))
        return

    made = []
    if "spectrum" in artifacts:
        made.append(plot.plot_spectrum(artifacts["spectrum"], out_dir / "spectrum.png"))
    if "calibration" in artifacts:
        p = plot.plot_calibration_tail(artifacts["calibration"], out_dir / "calibration_tail.png")
        if p:
            made.append(p)
    if "sweep" in artifacts:
        p = plot.plot_sweep(artifacts["sweep"], out_dir / "sweep.png")
        if p:
            made.append(p)
    if made:
        print("Wrote plots: " + ", ".join(str(p) for p in made))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a whitening YAML config")
    parser.add_argument("--out-dir", default=None, help="Override report.out_dir from the config")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    try:
        checks, meta, artifacts = run(cfg)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"whitening pre-flight could not run: {e}", file=sys.stderr)
        return 2

    report_cfg = cfg.get("report", {})
    out_dir = Path(args.out_dir or report_cfg.get("out_dir", "results/whitening"))
    paths = write_report(checks, meta, out_dir, stem=report_cfg.get("stem", "whitening_report"))
    print(paths["txt"].read_text())
    print(f"\nWrote {paths['json']} and {paths['txt']}")

    if report_cfg.get("plots", True):
        _make_plots(checks, artifacts, out_dir)

    overall = worst(checks)
    return 1 if overall == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
