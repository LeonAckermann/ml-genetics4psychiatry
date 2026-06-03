#!/usr/bin/env python3
"""
Stage-1 Miami plot: Schizophrenia GWAS (top) vs MRI Area_BA1_left GWAS (bottom, mirrored).
Ghost layers on the bottom panel suggest 1,010 MRI phenotypes processed in the pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
SCZ_PATH = BASE / "data/pipeline/input/gwas_illness/z_SCZ.txt"
MRI_PATH = BASE / "data/pipeline/input/gwas_mri/pheno_gwas/Area_BA1_left/allRES.txt"
OUT_PATH = BASE / "figures/miami_stage1.png"

# ── Constants ──────────────────────────────────────────────────────────────────
GWAS_SIG   = 5e-8
SUGGESTIVE = 1e-5
CHR_ORDER  = [str(i) for i in range(1, 23)]  # autosomes only

# Thinning only used for ghost layers so they don't dominate render time
GHOST_THIN_FRAC = 0.05

# Colours
C_ODD  = "#2166AC"   # deep blue  — odd chromosomes
C_EVEN = "#92C5DE"   # light blue — even chromosomes
C_SIG  = "#D6604D"   # red        — genome-wide significant hits
C_SUGG = "#FDAE61"   # amber      — suggestive threshold

# Ghost layers representing the 1,010 MRI phenotypes
N_MRI_PHENOTYPES = 1010
N_GHOSTS         = 4     # number of ghost layers rendered below the main MRI plot
GHOST_STEP       = 1.2   # logP units of depth per ghost layer
GHOST_ALPHA_BASE = 0.8  # alpha of the layer closest to the main plot


# ── Data loading ───────────────────────────────────────────────────────────────

def load_scz() -> pd.DataFrame:
    print("  Loading SCZ GWAS …")
    df = pd.read_csv(SCZ_PATH, sep="\t", usecols=["chrom", "pos", "P"],
                     low_memory=False)
    df.columns = ["CHR", "BP", "P"]
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"]).query("CHR in @CHR_ORDER").reset_index(drop=True)
    print(f"  → {len(df):,} SNPs")
    return df


def load_mri() -> pd.DataFrame:
    print("  Loading MRI GWAS …")
    df = pd.read_csv(MRI_PATH, sep="\t", usecols=["#CHROM", "POS", "P"],
                     low_memory=False)
    df.columns = ["CHR", "BP", "P"]
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"]).query("CHR in @CHR_ORDER").reset_index(drop=True)
    print(f"  → {len(df):,} SNPs")
    return df


# ── Genomic coordinates ────────────────────────────────────────────────────────

def add_cumpos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Append cumulative genomic x-position and −log10(P) columns."""
    df = df.copy()
    df["CHR"] = pd.Categorical(df["CHR"], categories=CHR_ORDER, ordered=True)
    df = df.sort_values(["CHR", "BP"])

    chr_max = df.groupby("CHR", observed=True)["BP"].max()
    GAP = 8_000_000  # 8 Mb gap between chromosomes on the x-axis

    offsets, cursor = {}, 0
    for ch in CHR_ORDER:
        if ch in chr_max.index:
            offsets[ch] = cursor
            cursor += int(chr_max[ch]) + GAP

    df = df[df["CHR"].isin(offsets)].copy()
    df["cumpos"] = df["BP"].values + df["CHR"].map(offsets).astype(np.int64).values
    df["logP"]   = -np.log10(df["P"].clip(lower=1e-300))
    return df, offsets


def chr_midpoints(df: pd.DataFrame) -> tuple[list, list]:
    ticks, labels = [], []
    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub):
            ticks.append(sub["cumpos"].mean())
            labels.append(ch)
    return ticks, labels


def chr_color(ch: str) -> str:
    n = int(ch) if ch.isdigit() else 23
    return C_ODD if n % 2 == 1 else C_EVEN


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_manhattan(ax, df: pd.DataFrame, direction: int = 1,
                   s: float = 0.4, alpha: float = 0.6) -> None:
    """Scatter plot with alternating chromosome colours."""
    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cumpos"], direction * sub["logP"],
                   c=chr_color(ch), s=s, alpha=alpha,
                   linewidths=0, rasterized=True)


def draw_ghost_layers(ax, df: pd.DataFrame) -> None:
    """
    Render N_GHOSTS semi-transparent copies of the MRI Manhattan plot below
    the main one. Uses a thinned subset so ghost layers don't dominate render time.
    Layers furthest from the main plot are drawn first (lowest z-order).
    """
    ghost_df = df.sample(frac=GHOST_THIN_FRAC, random_state=42)
    for k in range(N_GHOSTS, 0, -1):
        depth = k * GHOST_STEP
        alpha = GHOST_ALPHA_BASE * (1.0 - (k - 1) / N_GHOSTS)
        for ch in CHR_ORDER:
            sub = ghost_df[ghost_df["CHR"] == ch]
            if len(sub) == 0:
                continue
            ax.scatter(sub["cumpos"], -(sub["logP"] + depth),
                       c=chr_color(ch), s=1.2, alpha=alpha,
                       linewidths=0, rasterized=True)


# ── Main plot ──────────────────────────────────────────────────────────────────

def plot_miami() -> None:
    scz, _         = add_cumpos(load_scz())
    mri, mri_offs  = add_cumpos(load_mri())

    # Dynamic y-limits
    top_ymax  = scz["logP"].max() * 1.12
    bot_depth = -(mri["logP"].max() + N_GHOSTS * GHOST_STEP) * 1.08

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.04, "height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("white")

    # ── TOP: Schizophrenia ────────────────────────────────────────────────────
    draw_manhattan(ax_top, scz, direction=1)

    sig_scz = scz[scz["P"] < GWAS_SIG]
    if len(sig_scz):
        ax_top.scatter(sig_scz["cumpos"], sig_scz["logP"],
                       c=C_SIG, s=10, alpha=0.9, linewidths=0, zorder=6,
                       rasterized=True)

    ax_top.axhline(-np.log10(GWAS_SIG),   color=C_SIG,  lw=0.9, ls="--", alpha=0.75)
    ax_top.axhline(-np.log10(SUGGESTIVE), color=C_SUGG, lw=0.7, ls=":",  alpha=0.65)

    ax_top.set_ylim(0, top_ymax)
    ax_top.set_ylabel("−log₁₀(p)", fontsize=11)
    ax_top.set_title("Schizophrenia GWAS", loc="left", fontsize=13,
                     fontweight="bold", pad=8)
    ax_top.tick_params(axis="x", bottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    # ── BOTTOM: MRI (mirrored + ghost layers) ─────────────────────────────────
    draw_ghost_layers(ax_bot, mri)
    draw_manhattan(ax_bot, mri, direction=-1)

    sig_mri = mri[mri["P"] < GWAS_SIG]
    if len(sig_mri):
        ax_bot.scatter(sig_mri["cumpos"], -sig_mri["logP"],
                       c=C_SIG, s=10, alpha=0.9, linewidths=0, zorder=6,
                       rasterized=True)

    ax_bot.axhline(np.log10(GWAS_SIG),   color=C_SIG,  lw=0.9, ls="--", alpha=0.75)
    ax_bot.axhline(np.log10(SUGGESTIVE), color=C_SUGG, lw=0.7, ls=":",  alpha=0.65)

    # y-axis: display absolute values even though data is negative
    ax_bot.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{abs(y):.0f}")
    )
    ax_bot.set_ylim(bot_depth, 0)
    ax_bot.set_ylabel("−log₁₀(p)", fontsize=11)
    # Title at the center line with a white background so it sits cleanly
    #ax_bot.text(
    #    0.01, 1.0,
    #    f"MRI cortical area BA1 left  [1 of {N_MRI_PHENOTYPES:,} MRI phenotypes]",
    #    transform=ax_bot.transAxes,
    #    fontsize=11, fontweight="bold", va="bottom", ha="left",
    #    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
    #)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    # Ghost layer annotation
    ax_bot.annotate(
        f"·· {N_MRI_PHENOTYPES:,} MRI phenotypes",
        xy=(scz["cumpos"].max() * 0.97, bot_depth * 0.65),
        fontsize=8, color="grey", ha="right", style="italic",
    )

    # ── Shared x-axis (chromosome labels) ─────────────────────────────────────
    ticks, labels = chr_midpoints(scz)
    ax_bot.set_xticks(ticks)
    ax_bot.set_xticklabels(labels, fontsize=8)
    ax_bot.set_xlabel("Chromosome", fontsize=11)

    # x-range padded slightly
    xmin = scz["cumpos"].min() - 10e6
    xmax = scz["cumpos"].max() + 10e6
    ax_top.set_xlim(xmin, xmax)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=C_SIG,  ls="--", lw=1.2,
               label="Genome-wide significance  (p < 5×10⁻⁸)"),
        Line2D([0], [0], color=C_SUGG, ls=":",  lw=1.2,
               label="Suggestive threshold  (p < 10⁻⁵)"),
        Line2D([0], [0], ls="none", marker="o", markersize=5,
               markerfacecolor=C_SIG, markeredgewidth=0,
               label="Significant SNP"),
    ]
    ax_top.legend(handles=legend_handles, fontsize=8.5, frameon=False,
                  loc="upper right", handlelength=2)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    plot_miami()
