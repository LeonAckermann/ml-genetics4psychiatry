"""Plot experiment results: regression boxplots, binary classification line plots, regression line plots."""

import json
import glob
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
ILLNESSES = ["ADHD", "AZ", "BIP", "MDD", "OCD", "SCZ"]

Path(PLOTS_DIR).mkdir(exist_ok=True)

# Prettier model display names
MODEL_DISPLAY = {
    "lasso_regression": "Lasso",
    "ridge_regression": "Ridge",
    "linear_regression": "Linear",
    "residual_dnn": "Residual DNN",
    "tabpfn": "TabPFN",
    "xgboost": "XGBoost",
    "lasso_logistic_regression": "Lasso Logistic",
    "ridge_logistic_regression": "Ridge Logistic",
}


def parse_folder_name(folder):
    """Parse experiment folder name into components.

    Format: {model}_{illness}_p{p_value}_low_{row_ratio}_1_{type}
    Returns dict with model, illness, p_value, row_ratio, exp_type, or None on failure.
    """
    for illness in ILLNESSES:
        pattern = rf"^(.+)_{illness}_(p[\d.]+)_low_([\d.]+)_1_(regression|binary_classification)$"
        m = re.match(pattern, folder)
        if m:
            return {
                "model": m.group(1),
                "illness": illness,
                "p_value": float(m.group(2)[1:]),
                "row_ratio": float(m.group(3)),
                "exp_type": m.group(4),
            }
    return None


def load_all_results():
    regression = []
    binary = []

    for filepath in glob.glob(f"{RESULTS_DIR}/*/*.json"):
        folder = Path(filepath).parent.name
        meta = parse_folder_name(folder)
        if meta is None:
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            continue

        hpo = data.get("hpo", {})
        fold_metrics = hpo.get("fold_metrics", [])

        if meta["exp_type"] == "regression":
            entry = {
                **meta,
                "mean_pearson_r2": hpo.get("mean_pearson_r2"),
                "std_pearson_r2": hpo.get("std_pearson_r2"),
                "fold_pearson_r2": [fm.get("pearson_r2") for fm in fold_metrics],
                "mean_spearman_rho": hpo.get("mean_spearman_rho"),
                "std_spearman_rho": hpo.get("std_spearman_rho"),
                "fold_spearman_rho": [fm.get("spearman_rho") for fm in fold_metrics],
            }
            regression.append(entry)
        else:
            entry = {
                **meta,
                "mean_balanced_accuracy": hpo.get("mean_balanced_accuracy"),
                "std_balanced_accuracy": hpo.get("std_balanced_accuracy"),
                "fold_balanced_accuracy": [fm.get("balanced_accuracy") for fm in fold_metrics],
            }
            binary.append(entry)

    return pd.DataFrame(regression), pd.DataFrame(binary)


def make_model_palette(models):
    colors = sns.color_palette("tab10", n_colors=len(models))
    return dict(zip(models, colors))


# ── Plot 1: Regression boxplot (pearson r2 per model, per illness) ──────────

def plot1_regression_boxplot(df_reg):
    """Boxplot of pearson_r2 across folds for each model, per illness and p-value."""
    p_values = sorted(df_reg["p_value"].unique())

    for p_value in p_values:
        rows = []
        for _, row in df_reg[df_reg["p_value"] == p_value].iterrows():
            for val in row["fold_pearson_r2"]:
                if val is not None:
                    rows.append({
                        "illness": row["illness"],
                        "model": MODEL_DISPLAY.get(row["model"], row["model"]),
                        "pearson_r2": val,
                    })
        df = pd.DataFrame(rows)

        p_str = str(p_value)

        for illness in sorted(df["illness"].unique()):
            sub = df[df["illness"] == illness]
            models = sorted(sub["model"].unique())
            palette = make_model_palette(models)

            fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
            sns.boxplot(
                data=sub,
                x="model",
                y="pearson_r2",
                hue="model",
                order=models,
                hue_order=models,
                palette=palette,
                width=0.5,
                fliersize=3,
                legend=False,
                ax=ax,
            )
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"{illness} — Regression Performance (Pearson R², p={p_str})", fontsize=13, fontweight="bold")
            ax.set_xlabel("Model", fontsize=11)
            ax.set_ylabel("Pearson R²", fontsize=11)
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            out = f"{PLOTS_DIR}/plot1_regression_boxplot_{illness}_p{p_str}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")


# ── Plot 1b: Regression boxplot (Spearman ρ per model, per illness/p-value) ─

def plot1b_spearman_boxplot(df_reg):
    """Boxplot of Spearman ρ across folds for each model, per illness and p-value."""
    p_values = sorted(df_reg["p_value"].unique())

    for p_value in p_values:
        rows = []
        for _, row in df_reg[df_reg["p_value"] == p_value].iterrows():
            for val in row["fold_spearman_rho"]:
                if val is not None:
                    rows.append({
                        "illness": row["illness"],
                        "model": MODEL_DISPLAY.get(row["model"], row["model"]),
                        "spearman_rho": val,
                    })
        df = pd.DataFrame(rows)
        p_str = str(p_value)

        for illness in sorted(df["illness"].unique()):
            sub = df[df["illness"] == illness]
            models = sorted(sub["model"].unique())
            palette = make_model_palette(models)

            fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
            sns.boxplot(
                data=sub,
                x="model",
                y="spearman_rho",
                hue="model",
                order=models,
                hue_order=models,
                palette=palette,
                width=0.5,
                fliersize=3,
                legend=False,
                ax=ax,
            )
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"{illness} — Regression Performance (Spearman ρ, p={p_str})", fontsize=13, fontweight="bold")
            ax.set_xlabel("Model", fontsize=11)
            ax.set_ylabel("Spearman ρ", fontsize=11)
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            out = f"{PLOTS_DIR}/plot1b_spearman_boxplot_{illness}_p{p_str}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")


# ── Plot 2: Binary classification balanced accuracy vs row ratio (p=0.01) ───

def plot2_binary_row_ratio(df_bin):
    """Line plot of balanced accuracy vs row_ratio per p-value and illness."""
    p_values = sorted(df_bin["p_value"].unique())

    for p_value in p_values:
        sub = df_bin[df_bin["p_value"] == p_value].copy()
        if sub.empty:
            continue
        p_str = str(p_value)

        for illness in sorted(sub["illness"].unique()):
            illness_df = sub[sub["illness"] == illness]
            models = sorted(illness_df["model"].unique())
            palette = make_model_palette(models)

            fig, ax = plt.subplots(figsize=(8, 5))

            for model in models:
                mdf = illness_df[illness_df["model"] == model].sort_values("row_ratio")
                x = mdf["row_ratio"].values
                y = mdf["mean_balanced_accuracy"].values
                yerr = mdf["std_balanced_accuracy"].values
                label = MODEL_DISPLAY.get(model, model)
                color = palette[model]

                ax.plot(x, y, marker="o", label=label, color=color, linewidth=2, markersize=5)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance (0.5)")
            ax.set_title(f"{illness} — Binary Classification (p={p_str})\nBalanced Accuracy vs Row Ratio", fontsize=13, fontweight="bold")
            ax.set_xlabel("Row Ratio", fontsize=11)
            ax.set_ylabel("Balanced Accuracy", fontsize=11)
            ax.set_xticks(sorted(sub["row_ratio"].unique()))
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9, framealpha=0.8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            out = f"{PLOTS_DIR}/plot2_binary_row_ratio_{illness}_p{p_str}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")


# ── Plot 3: Regression pearson_r2 vs p-value ────────────────────────────────

def plot3_regression_pvalue(df_reg):
    """Line plot of mean pearson_r2 vs p-value per model, per illness."""
    for illness in sorted(df_reg["illness"].unique()):
        sub = df_reg[df_reg["illness"] == illness]
        models = sorted(sub["model"].unique())
        palette = make_model_palette(models)

        fig, ax = plt.subplots(figsize=(8, 5))

        for model in models:
            mdf = sub[sub["model"] == model].sort_values("p_value")
            x = mdf["p_value"].values
            y = mdf["mean_pearson_r2"].values
            yerr = mdf["std_pearson_r2"].values
            label = MODEL_DISPLAY.get(model, model)
            color = palette[model]

            ax.plot(x, y, marker="o", label=label, color=color, linewidth=2, markersize=5)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{illness} — Regression Performance\nPearson R² vs P-value Threshold", fontsize=13, fontweight="bold")
        ax.set_xlabel("P-value Threshold", fontsize=11)
        ax.set_ylabel("Pearson R²", fontsize=11)
        p_values = sorted(sub["p_value"].unique())
        ax.set_xticks(p_values)
        ax.set_xticklabels([str(p) for p in p_values])
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = f"{PLOTS_DIR}/plot3_regression_pvalue_{illness}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── Plot 3b: Regression Spearman ρ vs p-value ───────────────────────────────

def plot3b_spearman_pvalue(df_reg):
    """Line plot of mean Spearman ρ vs p-value per model, per illness."""
    for illness in sorted(df_reg["illness"].unique()):
        sub = df_reg[df_reg["illness"] == illness]
        models = sorted(sub["model"].unique())
        palette = make_model_palette(models)

        fig, ax = plt.subplots(figsize=(8, 5))

        for model in models:
            mdf = sub[sub["model"] == model].sort_values("p_value")
            x = mdf["p_value"].values
            y = mdf["mean_spearman_rho"].values
            yerr = mdf["std_spearman_rho"].values
            label = MODEL_DISPLAY.get(model, model)
            color = palette[model]

            ax.plot(x, y, marker="o", label=label, color=color, linewidth=2, markersize=5)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{illness} — Regression Performance\nSpearman ρ vs P-value Threshold", fontsize=13, fontweight="bold")
        ax.set_xlabel("P-value Threshold", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        p_values = sorted(sub["p_value"].unique())
        ax.set_xticks(p_values)
        ax.set_xticklabels([str(p) for p in p_values])
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = f"{PLOTS_DIR}/plot3b_spearman_pvalue_{illness}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading results...")
    df_reg, df_bin = load_all_results()
    print(f"  Regression experiments: {len(df_reg)}")
    print(f"  Binary classification experiments: {len(df_bin)}")

    print("\nPlot 1: Regression boxplots (pearson R² per model)...")
    plot1_regression_boxplot(df_reg)

    print("\nPlot 1b: Spearman ρ boxplots...")
    plot1b_spearman_boxplot(df_reg)

    print("\nPlot 2: Binary classification balanced accuracy vs row ratio (p=0.01)...")
    plot2_binary_row_ratio(df_bin)

    print("\nPlot 3: Regression pearson R² vs p-value...")
    plot3_regression_pvalue(df_reg)

    print("\nPlot 3b: Regression Spearman ρ vs p-value...")
    plot3b_spearman_pvalue(df_reg)

    print("\nDone! All plots saved to:", PLOTS_DIR)
