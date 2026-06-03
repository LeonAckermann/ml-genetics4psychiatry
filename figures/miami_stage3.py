#!/usr/bin/env python3
"""
Stage-2 Manhattan plots: SNPs present in BOTH the illness and MRI GWAS
(the aligned/clumped set).

Chromosome positions are looked up from the original GWAS files.
Outputs two matched plots (same size/scale as stage 1) for manual layering:
  figures/gwas_scz_aligned.png        — SCZ p-values, mirrored downward
  figures/gwas_mri_aligned.png        — MRI BA1-left p-values, upward
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE         = Path(__file__).parent.parent
ALIGNED_PATH = BASE / "data/pipeline/final/aligned_clumped_SCZ.txt"
SCZ_PATH     = BASE / "data/pipeline/input/gwas_illness/z_SCZ.txt"
MRI_PATH     = BASE / "data/pipeline/input/gwas_mri/pheno_gwas/Area_BA1_left/allRES.txt"

GWAS_SIG  = 5e-8
CHR_ORDER = [str(i) for i in range(1, 23)]

# ── Style (identical to stage 1) ──────────────────────────────────────────────
C_ODD   = "#9983B5"
C_EVEN  = "#5B8DB8"
C_SIG   = "#C0392B"
C_LINE  = "#444466"
C_FILL  = "#F0EAF8"

FIGSIZE = (7, 2.0)
DPI     = 200
S_MAIN  = 2.5     # larger dots — far fewer SNPs than stage 1
S_SIG   = 7.0
GAP_BP  = 8_000_000


# ── Data ───────────────────────────────────────────────────────────────────────

def load_aligned_ids() -> set:
    df = pd.read_csv(ALIGNED_PATH, sep="\t", usecols=["ID"])
    ids = set(df["ID"].astype(str))
    print(f"  Aligned SNPs: {len(ids):,}")
    return ids


def load_scz_aligned(ids: set) -> pd.DataFrame:
    print("  Joining SCZ …")
    df = pd.read_csv(SCZ_PATH, sep="\t",
                     usecols=["chrom", "pos", "rsID", "P"],
                     low_memory=False)
    df = df.rename(columns={"chrom": "CHR", "pos": "BP", "rsID": "ID"})
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df[df["ID"].isin(ids)].dropna(subset=["P"]).query("CHR in @CHR_ORDER")
    df = df.reset_index(drop=True)
    print(f"    {len(df):,} SNPs matched")
    return df


def load_mri_aligned(ids: set) -> pd.DataFrame:
    print("  Joining MRI …")
    df = pd.read_csv(MRI_PATH, sep="\t",
                     usecols=["#CHROM", "POS", "ID", "P"],
                     low_memory=False)
    df = df.rename(columns={"#CHROM": "CHR", "POS": "BP"})
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df[df["ID"].isin(ids)].dropna(subset=["P"]).query("CHR in @CHR_ORDER")
    df = df.reset_index(drop=True)
    print(f"    {len(df):,} SNPs matched")
    return df


# ── Coordinates ────────────────────────────────────────────────────────────────

def build_offsets(df: pd.DataFrame) -> dict:
    """Build cumulative x-offsets from the SCZ aligned set (reference coverage)."""
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


# ── Shared axis style ──────────────────────────────────────────────────────────

def _style(ax, ticks, labels, xmin, xmax):
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("−log₁₀(p)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)
    ax.tick_params(width=0.8, length=4, labelsize=9)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_mri(df: pd.DataFrame, ticks, labels, xmin, xmax, out: Path) -> None:
    """MRI aligned SNPs — upward."""
    sig_y = -np.log10(GWAS_SIG)
    ymax  = df["logP"].max() * 1.12

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    ax.axhspan(sig_y, ymax, color=C_FILL, zorder=0, lw=0)

    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cumpos"], sub["logP"],
                   c=chr_color(ch), s=S_MAIN, alpha=0.7,
                   linewidths=0, rasterized=True)

    sig = df[df["P"] < GWAS_SIG]
    if len(sig):
        ax.scatter(sig["cumpos"], sig["logP"],
                   c=C_SIG, s=S_SIG, alpha=0.9, linewidths=0,
                   zorder=6, rasterized=True)

    ax.axhline(sig_y, color=C_LINE, lw=0.8, ls="--", zorder=4)
    ax.set_ylim(0, ymax)
    _style(ax, ticks, labels, xmin, xmax)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_illness(df: pd.DataFrame, ticks, labels, xmin, xmax, out: Path) -> None:
    """SCZ aligned SNPs — upward (normal orientation)."""
    sig_y = -np.log10(GWAS_SIG)
    ymax  = df["logP"].max() * 1.12

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    ax.axhspan(sig_y, ymax, color=C_FILL, zorder=0, lw=0)

    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cumpos"], sub["logP"],
                   c=chr_color(ch), s=S_MAIN, alpha=0.7,
                   linewidths=0, rasterized=True)

    sig = df[df["P"] < GWAS_SIG]
    if len(sig):
        ax.scatter(sig["cumpos"], sig["logP"],
                   c=C_SIG, s=S_SIG, alpha=0.9, linewidths=0,
                   zorder=6, rasterized=True)

    ax.axhline(sig_y, color=C_LINE, lw=0.8, ls="--", zorder=4)
    ax.set_ylim(0, ymax)
    _style(ax, ticks, labels, xmin, xmax)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading aligned SNP IDs …")
    ids = load_aligned_ids()

    print("Matching against GWAS files …")
    scz_raw = load_scz_aligned(ids)
    mri_raw = load_mri_aligned(ids)

    # Shared coordinate system from SCZ (better chr coverage)
    offsets       = build_offsets(scz_raw)
    scz           = apply_coords(scz_raw, offsets)
    mri           = apply_coords(mri_raw, offsets)

    ticks, labels = chr_ticks(scz)
    xmin = scz["cumpos"].min() - 5e6
    xmax = scz["cumpos"].max() + 5e6

    print("Rendering …")
    plot_illness(scz, ticks, labels, xmin, xmax,
                 BASE / "figures/gwas_scz_aligned.png")
    plot_mri(mri, ticks, labels, xmin, xmax,
             BASE / "figures/gwas_mri_area_ba1_left_aligned.png")

    print("Done.")


if __name__ == "__main__":
    main()
