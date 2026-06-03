#!/usr/bin/env python3
"""
Layered Miami plot — deck-of-cards style.

Only the FRONT card shows real data (one MRI phenotype).
Behind it, N_GHOST_CARDS blank white rectangles are stacked, each
shifted down by GHOST_STEP logP-units and slightly to the right,
so their top edges peek out like a physical deck of cards.

Output: figures/miami_layered.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE     = Path(__file__).parent.parent
SCZ_PATH = BASE / "data/pipeline/input/gwas_illness/z_SCZ.txt"
MRI_PATH = BASE / "data/pipeline/input/gwas_mri/pheno_gwas/Area_BA1_right/allRES.txt"
OUT_PATH = BASE / "figures/miami_layered.png"

GWAS_SIG  = 5e-8
CHR_ORDER = [str(i) for i in range(1, 23)]

# ── Style ──────────────────────────────────────────────────────────────────────
C_ODD  = "#9983B5"
C_EVEN = "#5B8DB8"
C_SIG  = "#C0392B"
C_LINE = "#444466"
C_FILL = "#F0EAF8"

FIGSIZE = (8, 5.5)
DPI     = 200
S_MAIN  = 0.35
S_SIG   = 5.0
GAP_BP  = 8_000_000

# ── Deck parameters ────────────────────────────────────────────────────────────
N_GHOST_CARDS = 20     # blank cards stacked behind the real one
GHOST_STEP    = 1.2    # logP units each card peeks below the one in front
GHOST_X_STEP  = 8e6    # bp x-shift per card (perspective)
N_MRI_TOTAL   = 1010   # label only


# ── Data ───────────────────────────────────────────────────────────────────────

def load_scz() -> pd.DataFrame:
    print("  SCZ …")
    df = pd.read_csv(SCZ_PATH, sep="\t", usecols=["chrom", "pos", "P"],
                     low_memory=False)
    df.columns = ["CHR", "BP", "P"]
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"]).query("CHR in @CHR_ORDER").reset_index(drop=True)
    print(f"    {len(df):,} SNPs")
    return df


def load_mri() -> pd.DataFrame:
    print("  MRI …")
    df = pd.read_csv(MRI_PATH, sep="\t", usecols=["#CHROM", "POS", "P"],
                     low_memory=False)
    df.columns = ["CHR", "BP", "P"]
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"]).query("CHR in @CHR_ORDER").reset_index(drop=True)
    print(f"    {len(df):,} SNPs")
    return df


# ── Coordinates ────────────────────────────────────────────────────────────────

def build_offsets(df: pd.DataFrame) -> dict:
    chr_max = df.groupby("CHR")["BP"].max()
    offsets, cursor = {}, 0
    for ch in CHR_ORDER:
        if ch in chr_max.index:
            offsets[ch] = cursor
            cursor += int(chr_max[ch]) + GAP_BP
    return offsets


def apply_coords(df: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    df = df[df["CHR"].isin(offsets)].copy()
    df["cumpos"] = df["BP"].values + df["CHR"].map(offsets).astype(np.int64).values
    df["logP"]   = -np.log10(df["P"].clip(lower=1e-300))
    return df


def chr_ticks(df: pd.DataFrame) -> tuple[list, list]:
    ticks, labels = [], []
    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub):
            ticks.append(sub["cumpos"].mean())
            labels.append(ch)
    return ticks, labels


def chr_color(ch: str) -> str:
    return C_ODD if int(ch) % 2 == 1 else C_EVEN


# ── Axis style ─────────────────────────────────────────────────────────────────

def _style(ax, ticks, labels, xmin, xmax, xlabel=True):
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if xlabel else [""] * len(labels), fontsize=9)
    if xlabel:
        ax.set_xlabel("Chromosome position", fontsize=10)
    ax.set_ylabel("−log₁₀(p)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)
    ax.tick_params(width=0.8, length=4, labelsize=9)


# ── Deck drawing ───────────────────────────────────────────────────────────────

def draw_deck(ax, fg: pd.DataFrame, xmin: float, xmax: float) -> float:
    """
    Draw the foreground Manhattan + N_GHOST_CARDS blank cards behind it.

    Ghost cards are placed below the foreground (more negative y) and shifted
    slightly to the right so their top edges are visible as card edges.
    They contain no data — just a white rectangle with a thin border line.
    """
    max_logp = fg["logP"].max()
    sig_logp = -np.log10(GWAS_SIG)

    # Total x span including all ghost shifts
    x_total = xmax + N_GHOST_CARDS * GHOST_X_STEP

    # ── Ghost cards: drawn back-to-front ──────────────────────────────────────
    # Ghost k=N is furthest back, k=1 is just behind the foreground.
    # Baseline of ghost k = -(max_logp + k * GHOST_STEP)  [below the real card]
    for k in range(N_GHOST_CARDS, 0, -1):
        baseline = -(max_logp + k * GHOST_STEP)
        x_off    = k * GHOST_X_STEP
        z        = k   # back cards have low zorder, front ghosts have higher

        # White card body — occludes cards further behind in its y-range
        ax.fill_between(
            [xmin + x_off, x_total],
            baseline - max_logp,   # card bottom (never actually visible)
            baseline,              # card top = the visible edge
            color="white", zorder=z, lw=0,
        )

        # Card top-edge line (the only visible part of each ghost)
        ax.plot(
            [xmin + x_off, x_total],
            [baseline, baseline],
            color="#BBBBBB", lw=0.6, zorder=z + 1,
            solid_capstyle="round",
        )

    # ── Foreground card — highest zorder, covers ghost bodies ─────────────────
    Z = N_GHOST_CARDS * 2 + 20

    # White background for the foreground card
    ax.fill_between(
        [xmin, xmax],
        -max_logp * 1.08, 0,
        color="white", zorder=Z, lw=0,
    )

    # Lavender fill in the significance region
    ax.fill_between(
        [xmin, xmax],
        -max_logp * 1.08, -sig_logp,
        color=C_FILL, zorder=Z, lw=0,
    )

    # Manhattan scatter
    for ch in CHR_ORDER:
        sub = fg[fg["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cumpos"], -sub["logP"],
                   c=chr_color(ch), s=S_MAIN, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=Z + 1)

    # Significant hits
    sig = fg[fg["P"] < GWAS_SIG]
    if len(sig):
        ax.scatter(sig["cumpos"], -sig["logP"],
                   c=C_SIG, s=S_SIG, alpha=0.9, linewidths=0,
                   zorder=Z + 2, rasterized=True)

    # Significance threshold line
    ax.plot([xmin, xmax], [-sig_logp, -sig_logp],
            color=C_LINE, lw=0.8, ls="--", zorder=Z + 1)

    return -(max_logp * 1.08 + N_GHOST_CARDS * GHOST_STEP)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading …")
    scz_raw = load_scz()
    mri_raw = load_mri()

    offsets = build_offsets(scz_raw)
    scz     = apply_coords(scz_raw, offsets)
    mri     = apply_coords(mri_raw, offsets)

    ticks, labels = chr_ticks(scz)
    xmin = scz["cumpos"].min() - 5e6
    xmax = scz["cumpos"].max() + 5e6

    # x-limit for the bottom panel is wider (ghost cards shift right)
    xmax_bot = xmax + N_GHOST_CARDS * GHOST_X_STEP * 1.02

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=FIGSIZE,
        gridspec_kw={"hspace": 0.06, "height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("white")

    # ── TOP: SCZ ──────────────────────────────────────────────────────────────
    sig_y_top = -np.log10(GWAS_SIG)
    ymax_top  = scz["logP"].max() * 1.10
    ax_top.fill_between([xmin, xmax], sig_y_top, ymax_top,
                        color=C_FILL, zorder=0, lw=0)
    for ch in CHR_ORDER:
        sub = scz[scz["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax_top.scatter(sub["cumpos"], sub["logP"],
                       c=chr_color(ch), s=S_MAIN, alpha=0.55,
                       linewidths=0, rasterized=True)
    sig_scz = scz[scz["P"] < GWAS_SIG]
    if len(sig_scz):
        ax_top.scatter(sig_scz["cumpos"], sig_scz["logP"],
                       c=C_SIG, s=S_SIG, alpha=0.9, linewidths=0,
                       zorder=6, rasterized=True)
    ax_top.axhline(sig_y_top, color=C_LINE, lw=0.8, ls="--")
    ax_top.set_ylim(0, ymax_top)
    ax_top.set_title("Schizophrenia GWAS", loc="left", fontsize=10,
                     fontweight="bold", pad=5)
    _style(ax_top, ticks, labels, xmin, xmax, xlabel=False)
    ax_top.tick_params(axis="x", bottom=False)
    ax_top.spines["bottom"].set_visible(False)

    # ── BOTTOM: MRI deck ──────────────────────────────────────────────────────
    y_min = draw_deck(ax_bot, mri, xmin, xmax)

    # y-axis: label the foreground card's scale
    mri_max = int(mri["logP"].max()) + 2
    yticks  = list(range(0, -mri_max - 1, -5))
    ax_bot.set_yticks(yticks)
    ax_bot.set_yticklabels([str(abs(y)) for y in yticks], fontsize=9)
    ax_bot.set_ylim(y_min, 0)
    ax_bot.set_ylabel("−log₁₀(p)", fontsize=10)

    ax_bot.text(
        0.01, 1.0,
        f"MRI cortical area BA1 left  [1 of {N_MRI_TOTAL:,} MRI phenotypes]",
        transform=ax_bot.transAxes, fontsize=9, fontweight="bold",
        va="bottom", ha="left",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
    )

    _style(ax_bot, ticks, labels, xmin, xmax_bot, xlabel=True)
    ax_bot.spines["top"].set_visible(False)

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
