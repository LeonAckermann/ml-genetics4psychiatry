#!/usr/bin/env python3
"""
Two matched Manhattan plots for manual layering:
  figures/gwas_scz.png        — Schizophrenia (upward)
  figures/gwas_mri_ba1left.png — MRI Area BA1 left (mirrored / downward)

Both share identical figsize, DPI, and x-axis scale so they align
pixel-perfectly when stacked in Figma / Illustrator / etc.
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
MRI_PATH = BASE / "data/pipeline/input/gwas_mri/pheno_gwas/Mean_thickness_G-oc-temp-med-Lingual_right/allRES.txt"

GWAS_SIG  = 5e-8
CHR_ORDER = [str(i) for i in range(1, 23)]

# ── Style ──────────────────────────────────────────────────────────────────────
C_ODD   = "#9983B5"   # purple  — odd chromosomes
C_EVEN  = "#5B8DB8"   # blue    — even chromosomes
C_SIG   = "#C0392B"   # red     — significant SNPs
C_LINE  = "#444466"   # dark    — threshold dashed line
C_FILL  = "#F0EAF8"   # lavender — shaded region beyond threshold

FIGSIZE = (7, 2)    # width, height in inches — keep identical for both
DPI     = 200
S_MAIN  = 0.35        # scatter marker size (dense full-genome data)
S_SIG   = 5.0         # marker size for significant hits
GAP_BP  = 8_000_000   # gap between chromosomes on x-axis


# ── Data ───────────────────────────────────────────────────────────────────────

def load_scz() -> pd.DataFrame:
    print("  Loading SCZ …")
    df = pd.read_csv(SCZ_PATH, sep="\t", usecols=["chrom", "pos", "P"],
                     low_memory=False)
    df.columns = ["CHR", "BP", "P"]
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["P"]   = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"]).query("CHR in @CHR_ORDER").reset_index(drop=True)
    print(f"    {len(df):,} SNPs")
    return df


def load_mri() -> pd.DataFrame:
    print("  Loading MRI …")
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
    """Chr → cumulative bp offset (derived from SCZ as reference)."""
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


# ── Shared axis styling ────────────────────────────────────────────────────────

def _style(ax, ticks, labels, xmin, xmax, xlabel=True):
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if xlabel else [""] * len(labels), fontsize=9)
    ax.set_ylabel("−log₁₀(p)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)
    ax.tick_params(width=0.8, length=4, labelsize=9)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_mri(df: pd.DataFrame, ticks, labels, xmin, xmax, out: Path) -> None:
    """Brain phenotype — upward (normal orientation)."""
    sig_y = -np.log10(GWAS_SIG)
    ymax  = df["logP"].max() * 1.10

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    ax.axhspan(sig_y, ymax, color=C_FILL, zorder=0, lw=0)

    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cumpos"], sub["logP"],
                   c=chr_color(ch), s=S_MAIN, alpha=0.55,
                   linewidths=0, rasterized=True)

    sig = df[df["P"] < GWAS_SIG]
    if len(sig):
        ax.scatter(sig["cumpos"], sig["logP"],
                   c=C_SIG, s=S_SIG, alpha=0.9, linewidths=0,
                   zorder=6, rasterized=True)

    ax.axhline(sig_y, color=C_LINE, lw=0.8, ls="--", zorder=4)
    ax.set_ylim(0, ymax)
    _style(ax, ticks, labels, xmin, xmax)
    ax.tick_params(axis='x', labelbottom=False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_illness(df: pd.DataFrame, ticks, labels, xmin, xmax, out: Path) -> None:
    """Illness GWAS — mirrored (downward)."""
    sig_y = np.log10(GWAS_SIG)   # negative
    ymin  = -(df["logP"].max() * 1.10)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    ax.axhspan(ymin, sig_y, color=C_FILL, zorder=0, lw=0)

    for ch in CHR_ORDER:
        sub = df[df["CHR"] == ch]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cumpos"], -sub["logP"],
                   c=chr_color(ch), s=S_MAIN, alpha=0.55,
                   linewidths=0, rasterized=True)

    sig = df[df["P"] < GWAS_SIG]
    if len(sig):
        ax.scatter(sig["cumpos"], -sig["logP"],
                   c=C_SIG, s=S_SIG, alpha=0.9, linewidths=0,
                   zorder=6, rasterized=True)

    ax.axhline(sig_y, color=C_LINE, lw=0.8, ls="--", zorder=4)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{abs(y):.0f}")
    )
    ax.set_ylim(ymin, 0)
    _style(ax, ticks, labels, xmin, xmax)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelbottom=False, labeltop=True, labelsize=9)
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_linewidth(0.7)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    scz_raw = load_scz()
    mri_raw = load_mri()

    # Shared coordinate system derived from SCZ coverage
    offsets      = build_offsets(scz_raw)
    scz          = apply_coords(scz_raw, offsets)
    mri          = apply_coords(mri_raw, offsets)

    ticks, labels = chr_ticks(scz)
    xmin = scz["cumpos"].min() - 5e6
    xmax = scz["cumpos"].max() + 5e6

    print("Rendering …")
    plot_illness(scz, ticks, labels, xmin, xmax,
                 BASE / "figures/gwas_scz.png")
    plot_mri(mri, ticks, labels, xmin, xmax,
             BASE / "figures/gwas_mri_Mean_thickness_G-oc-temp-med-Lingual_right.png")

    print("Done.")


if __name__ == "__main__":
    main()
