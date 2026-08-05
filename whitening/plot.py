"""Diagnostic plots. Kept separate from the checks so a headless/no-matplotlib
environment can still run the numeric pipeline; ``run.py`` calls this only
when ``report.plots: true`` (default) and swallows an ImportError with a
warning check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_spectrum(spec: dict, out_path: str | Path) -> Path:
    """log(lambda_i) vs i, per check 5: clean bulk + outliers vs smooth decay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = np.asarray(spec["eigenvalues"])[::-1]  # descending, i = 1..K
    i = np.arange(1, len(lam) + 1)
    pos = lam > 0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(i[pos], lam[pos], "o-", ms=3, color="#3b6ea5", label="lambda_i > 0")
    if (~pos).any():
        ax.scatter(i[~pos], np.abs(lam[~pos]), marker="x", color="#c0392b",
                   label="|lambda_i|, lambda_i <= 0", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("index i (descending eigenvalue order)")
    ax.set_ylabel("lambda_i (log scale)")
    ax.set_title(f"Spectrum: kappa={spec['condition_number']:.3g}, "
                 f"eff. rank={spec['effective_rank']:.1f}/{spec['K']}")
    ax.axvline(spec["gap_index"] + 0.5, color="gray", ls="--", lw=1,
               label=f"largest bottom-half gap (i={spec['gap_index']})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_calibration_tail(metrics: dict, out_path: str | Path) -> Path:
    """Empirical vs chi^2_K quantiles of |w|^2, log-log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = metrics["tail"]
    if not tail:
        return None
    qs = sorted(tail)
    emp = [tail[q]["empirical"] for q in qs]
    ref = [tail[q]["chi2_reference"] for q in qs]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ref, emp, "o-", color="#3b6ea5")
    lo, hi = min(ref + emp), max(ref + emp)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="perfect calibration")
    ax.set_xlabel("chi^2_K reference quantile")
    ax.set_ylabel("empirical |w|^2 quantile")
    ax.set_title("Tail calibration of whitened simulated null")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sweep(sweep: dict, out_path: str | Path) -> Path:
    """Alpha vs quad-form inflation and tail ratio, for the regularisation sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in sweep["results"] if r.get("feasible")]
    if not rows:
        return None
    alphas = [r["alpha"] for r in rows]
    infl = [r["quad_form_inflation"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, infl, "o-", color="#3b6ea5")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    if sweep.get("recommended_alpha") is not None:
        ax.axvline(sweep["recommended_alpha"], color="#c0392b", ls=":", lw=1.5,
                   label=f"recommended alpha={sweep['recommended_alpha']:g}")
    ax.set_xscale("log") if all(a > 0 for a in alphas) else None
    ax.set_xlabel(f"{sweep['method']} alpha")
    ax.set_ylabel("E[|w|^2] / K")
    ax.set_title("Regularisation sweep (null calibration)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
