"""Entry point for running experiments from a YAML config file.

Usage:
    python main.py --config experiments/linear_regression_scz.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from dataloader import load_illness_data
from dataloader.pipeline import (
    aligne_clumped_illness_mri,
    aligne_clumped_phenotype,
    aligne_illness_mri,
    call_plink2,
    construct_gwas_mri,
    construct_gwas_phenotype,
    included_phenotype_columns,
    prepare_phenotype_clump_input,
    select_dense_features,
)
from dataloader.preprocess import drop_target_correlated_features, sample
from src import get_default_search_space, nested_cv


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Model family → concrete model name
# ---------------------------------------------------------------------------

# Supported family names:
#   linear      → linear_regression      / logistic_regression
#   lasso       → lasso_regression       / lasso_logistic_regression
#   ridge       → ridge_regression       / ridge_logistic_regression
#   bayesian_ridge → bayesian_ridge_regression  (regression only)
#   xgboost     → xgboost               (same for both task types)
#   residual_dnn→ residual_dnn           (same for both task types)
#   tabpfn      → tabpfn                 (same for both task types)
#
# Families whose concrete name is identical for both task types fall through
# to the default in resolve_model_name and are returned unchanged.
_MODEL_NAME_MAP: dict[tuple[str, str], str] = {
    ("linear", "regression"):            "linear_regression",
    ("linear", "binary_classification"): "logistic_regression",
    ("lasso",  "regression"):            "lasso_regression",
    ("lasso",  "binary_classification"): "lasso_logistic_regression",
    ("ridge",  "regression"):            "ridge_regression",
    ("ridge",  "binary_classification"): "ridge_logistic_regression",
    # Regression-only. The binary entry maps to the same name so build_model
    # can raise a specific error instead of a bare "Unknown model".
    ("bayesian_ridge", "regression"):            "bayesian_ridge_regression",
    ("bayesian_ridge", "binary_classification"): "bayesian_ridge_regression",
}


def save_construction_stats(name: str, stats: dict) -> Path:
    """Save a one-time pipeline construction step's stats to results/<name>/."""
    results_dir = Path("./results") / name
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{name}_{timestamp}.json"
    with open(results_file, "w") as fh:
        json.dump(stats, fh, indent=2, cls=NumpyEncoder)
    print(f"Saved construction stats to {results_file}")
    return results_file


def resolve_model_name(family: str, task_type: str) -> str:
    """Return the concrete model name for a (family, task_type) pair.

    ``linear``, ``lasso``, and ``ridge`` resolve to different names depending
    on the task type.  ``xgboost``, ``residual_dnn``, and ``tabpfn`` are
    returned unchanged (same model handles both task types).
    """
    return _MODEL_NAME_MAP.get((family, task_type), family)


# ---------------------------------------------------------------------------
# Best-params loader
# ---------------------------------------------------------------------------

def load_best_params_from_folder(
    illness: str,
    p_clump,
    distribution: str,
    model_name: str,
    best_params_folder: str = "best_params",
) -> list | None:
    """Load fold best-params from the most recent matching JSON in best_params/."""
    import glob

    pattern = f"{best_params_folder}/**/{model_name}_{illness}_p{p_clump}_{distribution}*.json"
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None

    latest = files[-1]
    print(f"Loading best params from {latest}")
    with open(latest) as fh:
        data = json.load(fh)

    if "hpo" in data and "fold_best_params" in data["hpo"]:
        return data["hpo"]["fold_best_params"]
    raise ValueError(f"No fold_best_params found in {latest}")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def pipeline(cfg: dict) -> None:
    """Run data processing pipeline steps based on config."""
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML genetics experiments from a YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML file")
    parser.add_argument(
        "--load-best-params",
        type=str,
        default=None,
        help="Path to HPO result JSON to load fold best-params from (skips HPO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    plink_cfg = cfg["plink2"]
    construct_cfg = cfg.get("construct_gwas_mri", {})
    gwas_phenotype_cfg = cfg.get("gwas_phenotype_construction", {})
    phenotype_clumping_cfg = cfg.get("phenotype_clumping", {})
    data_cfg = cfg["data"]
    total_chunks = plink_cfg.get("total_chunks", None)
    chunk_size = plink_cfg.get("chunk_size", 10000)

    output: dict = {}

    # ── One-time GWAS MRI construction ───────────────────────────────────────
    if construct_cfg.get("run", False):
        print("\nConstructing merged GWAS MRI file...")
        stats = construct_gwas_mri(
            path=construct_cfg["input_path"],
            output_path=construct_cfg["output_path"],
            chunk_size=construct_cfg.get("chunk_size", 10000),
            total_chunks=construct_cfg.get("total_chunks", None),
            polars=construct_cfg.get("polars", False),
            value=construct_cfg.get("value", "T_STAT"),
        )
        output["gwas_mri_stats"] = stats
        save_construction_stats("construct_gwas_mri", stats)

    # ── One-time GWAS phenotype construction ─────────────────────────────────
    if gwas_phenotype_cfg.get("run", False):
        print("\nConstructing joined GWAS phenotype file...")
        stats = construct_gwas_phenotype(
            input_path=gwas_phenotype_cfg["input_path"],
            output_path=gwas_phenotype_cfg["output_path"],
            how=gwas_phenotype_cfg.get("how", "inner"),
            join_key=gwas_phenotype_cfg.get("join_key", "chrom"),
            info_csv_path=gwas_phenotype_cfg.get("info_csv_path"),
        )
        output["gwas_phenotype_stats"] = stats
        save_construction_stats("gwas_phenotype_construction", stats)

    # ── Per-phenotype clumping (intermediate/ + output/ only, no 'final' merge
    # unless create_final is set) ────────────────────────────────────────────
    if phenotype_clumping_cfg.get("run", False):
        print("\nRunning per-phenotype clumping...")
        gwas_pheno_path = phenotype_clumping_cfg["gwas_pheno_path"]
        phenotypes = phenotype_clumping_cfg.get("phenotypes")
        if not phenotypes:
            min_density = phenotype_clumping_cfg.get("min_density")
            if min_density is not None:
                density_json = phenotype_clumping_cfg.get(
                    "density_json", "./data/pipeline/analysis/dense_density.json")
                phenotypes = select_dense_features(density_json, float(min_density))
                print(f"Selected {len(phenotypes)} phenotypes at density >= "
                      f"{min_density} from {density_json}")
            else:
                phenotypes = included_phenotype_columns(
                    gwas_pheno_path, phenotype_clumping_cfg["info_csv_path"],
                )
        print(f"Phenotypes: {', '.join(phenotypes)}")

        create_final = phenotype_clumping_cfg.get("create_final", False)
        phenotype_stats = {}
        for phenotype in phenotypes:
            print(f"\n--- Phenotype: {phenotype} ---")
            prep_stats = prepare_phenotype_clump_input(
                phenotype=phenotype, gwas_pheno_path=gwas_pheno_path,
            )
            plink2 = {
                "--bfile": phenotype_clumping_cfg["ref"],
                "--clump": f"./data/pipeline/intermediate/aligned_{phenotype}.txt",
                "--clump-kb": phenotype_clumping_cfg.get("clump_kb", 500),
                "--clump-r2": float(phenotype_clumping_cfg.get("r2", 0.05)),
                "--clump-p1": phenotype_clumping_cfg.get("p_clump", 1),
                "--clump-p2": phenotype_clumping_cfg.get("p_clump", 1),
                "--out": f"./data/pipeline/output/clumped_{phenotype}",
            }
            call_plink2(plink2)
            stats = {"prepare": prep_stats, "plink2": plink2}
            if create_final:
                stats["final"] = aligne_clumped_phenotype(
                    phenotype=phenotype, gwas_pheno_path=gwas_pheno_path,
                )
            phenotype_stats[phenotype] = stats

        output["phenotype_clumping_stats"] = phenotype_stats
        save_construction_stats("phenotype_clumping", phenotype_stats)

    # ── Data pipeline ─────────────────────────────────────────────────────────
    if plink_cfg.get("prepare", False):
        mri_path = plink_cfg.get("mri", None)
        output_suffix = plink_cfg.get("output_suffix", "")
        print("\nRunning data processing pipeline...")
        first_alignment = aligne_illness_mri(
            illness=data_cfg["illness"], verbose=True, chunk_size=chunk_size,
            total_chunks=total_chunks, mri_path=mri_path,
            polars=plink_cfg.get("polars", False),
            output_suffix=output_suffix,
        )
        plink2 = {
            "--bfile": plink_cfg["ref"],
            "--clump": plink_cfg["aligned"],
            "--clump-kb": plink_cfg["clump_kb"],
            "--clump-r2": float(plink_cfg["r2"]),
            "--clump-p1": plink_cfg["p_clump"],
            "--clump-p2": plink_cfg["p_clump"],
            "--out": plink_cfg["output"],
        }
        call_plink2(plink2)
        output.update({
            "illness_mri_alignment": first_alignment,
            "plink2": plink2,
        })
        if plink_cfg.get("create_final", True):
            second_alignment = aligne_clumped_illness_mri(
                illness=data_cfg["illness"], verbose=True,
                polars=plink_cfg.get("polars", False),
                mri_path=mri_path, chunk_size=chunk_size, total_chunks=total_chunks,
                output_suffix=output_suffix,
            )
            output["clumped_illness_mri_alignment"] = second_alignment

    # ── Resolve HPO config ────────────────────────────────────────────────────
    hpo_cfg: dict = {}
    if isinstance(cfg.get("hpo"), dict):
        hpo_cfg = cfg["hpo"]
    hpo_enabled = cfg.get("hpo") is True or (hpo_cfg and hpo_cfg.get("run", True) is not False)

    load_best_params_file = args.load_best_params
    load_best_params_from_config = cfg.get("load_best_params", False)

    outer_cv = hpo_cfg.get("outer_cv", data_cfg.get("n_splits", 5))
    inner_cv = hpo_cfg.get("inner_cv", 3)

    if not (cfg["experiment"].get("run", True) or hpo_enabled or load_best_params_file):
        print("Experiment run flag is False — skipping training and evaluation.")
        return

    # ── Per-experiment loop ───────────────────────────────────────────────────
    # cfg["model"]["types"] allows looping over multiple task types in one run.
    # Falls back to the single cfg["model"]["type"] for backward compatibility.
    model_task_types: list[str] = cfg["model"].get(
        "types", [cfg["model"].get("type", "regression")]
    )

    # Noise levels: cfg["noise"]["sigma"] may be a scalar or a list.
    _noise_cfg = cfg.get("noise", {}) or {}
    _raw_sigma = _noise_cfg.get("sigma", [0.0])
    noise_levels: list[float] = (
        [float(_raw_sigma)] if not isinstance(_raw_sigma, list) else [float(s) for s in _raw_sigma]
    )

    # Random row fractions: data.rand may be a scalar or a list.
    _raw_rand = data_cfg.get("rand", [1.0])
    rand_fracs: list[float] = (
        [float(_raw_rand)] if not isinstance(_raw_rand, list) else [float(r) for r in _raw_rand]
    )

    # PCA settings: data.pca may be a scalar, list, or null.
    # Accepted values: float (0,1) for variance threshold, "effective", null.
    _raw_pca = data_cfg.get("pca", [None])
    if not isinstance(_raw_pca, list):
        _raw_pca = [_raw_pca]
    def _parse_pca(v):
        if v is None:
            return None
        if str(v).lower() == "effective":
            return "effective"
        return float(v)
    pca_values: list[float | str | None] = [_parse_pca(v) for v in _raw_pca]

    # data.illness left blank/empty → auto-detect the phenotype list. Prefer the
    # density-frontier selection (same features used for clumping) when
    # data.min_density is set; otherwise fall back to every include=1 phenotype
    # from info.csv that's present as a column in the gwas_pheno matrix.
    illness_list = data_cfg.get("illness") or []
    if not illness_list:
        min_density = data_cfg.get("min_density")
        if min_density is not None:
            density_json = data_cfg.get(
                "density_json", "./data/pipeline/analysis/dense_density.json")
            illness_list = select_dense_features(density_json, float(min_density))
            print(f"data.illness is empty — selected {len(illness_list)} phenotypes "
                  f"at density >= {min_density} from {density_json}: "
                  f"{', '.join(illness_list)}")
        else:
            illness_list = included_phenotype_columns(
                data_cfg["gwas_pheno_path"], data_cfg["info_csv_path"],
            )
            print(f"data.illness is empty — auto-detected {len(illness_list)} phenotypes "
                  f"from info.csv: {', '.join(illness_list)}")

    for dist, p, illness, row_ratio, col_ratio, task_type, noise_sigma, rand_frac, pca_var in product(
        data_cfg.get("distribution", []),
        data_cfg.get("p_clump", []),
        illness_list,
        data_cfg.get("row_ratio", [1.0]),
        data_cfg.get("col_ratio", [1.0]),
        cfg["model"].get("type", ["regression"]),
        noise_levels,
        rand_fracs,
        pca_values,
    ):
        model_family = cfg["model"]["name"]
        model_name = resolve_model_name(model_family, task_type)

        # Per-iteration cfg copy: override model.type, noise.sigma, and
        # data.pca_components so downstream calls see the correct values.
        iter_cfg = {
            **cfg,
            "model": {**cfg["model"], "type": task_type},
            "noise": {**(_noise_cfg), "sigma": noise_sigma},
            "data":  {**data_cfg, "pca": pca_var},
        }

        noise_suffix = f"_noise{noise_sigma:g}" if noise_sigma > 0 else ""
        rand_suffix  = f"_rand{rand_frac:g}"    if rand_frac  < 1.0 else ""
        if pca_var is None:
            pca_suffix = ""
        elif pca_var == "effective":
            pca_suffix = "_pcaeff"
        else:
            pca_suffix = f"_pca{int(float(pca_var) * 100)}"
        print(
            f"\nStarting experiment: illness={illness}, p_clump={p},"
            f" distribution={dist}, task_type={task_type}, model={model_name}"
            + (f", noise_sigma={noise_sigma:g}"          if noise_sigma > 0    else "")
            + (f", rand={rand_frac:g}"                   if rand_frac  < 1.0   else "")
            + (f", pca={pca_var}"                        if pca_var is not None else "")
        )
        experiment_name = f"{model_name}_{illness}_p{p}_{dist}_{row_ratio}_{col_ratio}_{task_type}{noise_suffix}{rand_suffix}{pca_suffix}"
        results_dir = Path("./results") / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{experiment_name}_{timestamp}.json"
        output = {}

        # ── Optional sampling ─────────────────────────────────────────────────
        if data_cfg.get("sampling", False):
            print(f"Sampling data for illness={illness}, p_clump={p}, distribution={dist}...")
            sampling_metrics = sample(
                p_value=p, distribution=dist, illness=illness,
                polars=data_cfg.get("polars", False),
                chunk_size=data_cfg.get("chunk_size", 100000),
                total_chunks=data_cfg.get("total_chunks", None),
                sample_p=data_cfg.get("sample_p", False),
                gwas_pheno_path=data_cfg.get("gwas_pheno_path"),
                clumps_path=data_cfg.get("clumps_path"),
                max_col_missing=data_cfg.get("max_col_missing"),
                min_complete_frac=data_cfg.get("min_complete_frac"),
                max_target_corr=data_cfg.get("max_target_corr", 0.8),
                use_parquet=data_cfg.get("use_parquet", True),
            )
            output[f"sampling_metrics_{illness}_{dist}_p{p}"] = sampling_metrics
        else:
            data_path = Path(f"./data/sampled/{dist}/sampled_{illness}_p{p}.txt")
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Sampled data not found at {data_path}. "
                    "Run with sampling: true first."
                )
            # The feature-pruning knobs all live inside sample(); with sampling
            # off they are silently inert and the pre-existing sampled file is
            # used as-is (whatever pruning it was written with). Say so, rather
            # than letting the config look like it applied.
            inert = [k for k in ("max_target_corr", "min_complete_frac", "max_col_missing")
                     if data_cfg.get(k) is not None]
            if inert:
                print(f"  NOTE: sampling is off — {', '.join(inert)} not applied; "
                      f"reusing {data_path} as written. Set data.sampling: true "
                      f"to recompute (and to report target correlations).")

        # ── Load data ─────────────────────────────────────────────────────────
        df = load_illness_data(
            illness,
            in_notebook=False,
            polars=data_cfg.get("polars", True),
            distribution=dist,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            p_value=p,
            row_ratio=row_ratio,
            col_ratio=col_ratio,
            top_rows=data_cfg.get("top_rows", True),
            top_cols=data_cfg.get("top_cols", True),
            mri_p_value=data_cfg.get("mri_p_value", 0.05),
        )
        print(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"{row_ratio*100}% of rows, {col_ratio*100}% of columns retained after sampling")
        print(f"Original shape {int(df.shape[0] / row_ratio)} samples, {int(df.shape[1] / col_ratio)} features")

        df_pandas = df.to_pandas() if hasattr(df, "to_pandas") else df

        # ── Marginal correlation with the target ──────────────────────────────
        # Applied to whatever matrix is about to be trained on, so it also
        # covers sampling: false (sampled files on disk predate this filter).
        # Before drop_missing: correlations use every pairwise-complete row, and
        # dropping a sparse feature here saves the rows it would have cost.
        # When sampling ran, this is a no-op re-check — the report then simply
        # describes the final training matrix.
        target_corr_stats = None
        max_target_corr = data_cfg.get("max_target_corr", 0.8)
        if max_target_corr is not None:
            df_pandas, target_corr_stats = drop_target_correlated_features(
                df_pandas, float(max_target_corr), target=data_cfg["target"],
            )
            print(f"  target correlation (max_target_corr={max_target_corr}): kept "
                  f"{target_corr_stats['n_features_kept']}/{target_corr_stats['n_features_before']} "
                  f"features ({target_corr_stats['n_features_dropped']} dropped"
                  + (f": {', '.join(target_corr_stats['dropped_columns'])}"
                     if target_corr_stats["dropped_columns"] else "") + ")")

        # ── Optional: drop rows with any missing value (complete-case matrix) ──
        # Feature columns (other phenotypes) can be null at a phenotype's selec
        # SNPs; enable data.drop_missing to train on a fully non-null matrix.
        drop_missing_stats = None
        if data_cfg.get("drop_missing", False):
            n_before = len(df_pandas)
            df_pandas = df_pandas.dropna().reset_index(drop=True)
            n_after = len(df_pandas)
            drop_missing_stats = {
                "n_before": n_before,
                "n_after": n_after,
                "n_dropped": n_before - n_after,
                "frac_dropped": (n_before - n_after) / n_before if n_before else 0.0,
            }
            print(f"  drop_missing: kept {n_after:,}/{n_before:,} rows with "
                  f"no missing values ({n_before - n_after:,} dropped)")

        id_cols = [col for col in ["ID"] if col in df_pandas.columns]
        X = df_pandas.drop(columns=[data_cfg["target"]] + id_cols)
        y = df_pandas[data_cfg["target"]]

        # Predictor features actually used from the sampled file (i.e. the
        # phenotype columns that survived the sampling-time column pruning).
        selected_features = {
            "n_features": int(X.shape[1]),
            "features": list(X.columns),
        }

        # ── Optional random row subsampling (distinct from row_ratio) ─────────
        if rand_frac < 1.0:
            n_total  = len(X)
            n_sample = max(1, int(n_total * rand_frac))
            rng      = np.random.default_rng(42)
            idx      = np.sort(rng.choice(n_total, size=n_sample, replace=False))
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            print(f"  Random subsample: {n_sample:,}/{n_total:,} rows (rand={rand_frac:g})")

        # ── Optional sign-flip of negative labels + their features ────────────
        if data_cfg.get("invert", False):
            neg = (y < 0).values
            y = y.copy()
            y.iloc[neg] *= -1
            X = X.copy()
            X.iloc[neg] *= -1
            print(f"  Inverted {neg.sum()} samples with negative labels")

        if task_type == "binary_classification" and model_family != "mdn":
            from scipy.stats import norm
            y = norm.sf(abs(y)) * 2
            y = (y <= iter_cfg["model"]["p_value_binary"]).astype(int)

        # ── Resolve best params and n_trials ──────────────────────────────────
        best_params_for_eval = None

        if load_best_params_from_config:
            try:
                best_params_for_eval = load_best_params_from_folder(
                    illness=illness, p_clump=p, distribution=dist, model_name=model_name,
                )
                if best_params_for_eval:
                    print(f"Loaded {len(best_params_for_eval)} fold params from best_params/")
            except Exception as e:
                print(f"Could not load pre-trained params: {e}")

        if load_best_params_file:
            print(f"Loading best params from {load_best_params_file}...")
            with open(load_best_params_file) as fh:
                loaded = json.load(fh)
            if "hpo" in loaded and "fold_best_params" in loaded["hpo"]:
                best_params_for_eval = loaded["hpo"]["fold_best_params"]
                print(f"Loaded {len(best_params_for_eval)} fold params from file")
            else:
                raise ValueError(f"No fold_best_params found in {load_best_params_file}")

        # determine number of trials
        if best_params_for_eval is not None:
            # Evaluate with pre-loaded params — no optimisation
            n_trials = 0
        elif hpo_enabled:
            # Run HPO; models without a search space fall back to n_trials=0
            n_trials = hpo_cfg.get("n_trials", 100)
            if not get_default_search_space(model_name, task_type):
                n_trials = 0
        else:
            # Plain evaluation using the hyperparameters from the config file
            best_params_for_eval = [dict(iter_cfg["model"])] * outer_cv
            n_trials = 0

        # ── Single nested CV call covers HPO + evaluation ─────────────────────
        results = nested_cv(
            X, y,
            model_name=model_name,
            cfg=iter_cfg,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            n_trials=n_trials,
            search_space=hpo_cfg.get("search_space"),
            best_params_list=best_params_for_eval,
            experiment_name=experiment_name,
        )

        output.update({
            "experiment": experiment_name,
            "noise_sigma": noise_sigma,
            "rand_frac": rand_frac,
            "pca": pca_var,
            "target_correlation": target_corr_stats,
            "drop_missing": drop_missing_stats,
            "selected_features": selected_features,
            "config": iter_cfg,
            "timestamp": timestamp,
            "hpo": results,
        })

        with open(results_file, "w") as fh:
            json.dump(output, fh, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
