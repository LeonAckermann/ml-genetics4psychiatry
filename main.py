"""Entry point for running experiments from a YAML config file.

Usage:
    python main.py --config experiments/linear_regression_scz.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError

from dataloader import load_illness_data, load_txt, load_txt_polars
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
from dataloader.preprocess import drop_same_category_features, drop_target_correlated_features, sample
from src import RootConfig, get_default_search_space, nested_cv


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
# Config validation
# ---------------------------------------------------------------------------

def validate_config(cfg: dict, config_path: Path) -> RootConfig:
    """Validate the entire config against src.config.RootConfig.

    Catches typo'd or wrong-type keys at startup instead of a `dict.get(...)`
    call somewhere deep in the pipeline silently ignoring them, and exits with
    a readable message on failure rather than a raw pydantic traceback.

    Returns the validated model so callers can read the handful of values
    that have been migrated to use it directly (see the "Per-experiment loop"
    setup in main()); everything else still reads the original `cfg` dict --
    see src/config.py's module docstring for why that migration is
    deliberately partial for now.
    """
    try:
        return RootConfig(**cfg)
    except ValidationError as e:
        raise SystemExit(f"Invalid config {config_path}:\n\n{e}")


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

    validated = validate_config(cfg, config_path)

    plink_cfg = cfg.get("plink2", {})
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

    outer_cv = hpo_cfg.get("outer_cv", 5)
    inner_cv = hpo_cfg.get("inner_cv", 5)

    if not (cfg["experiment"].get("run", True) or hpo_enabled or load_best_params_file):
        print("Experiment run flag is False — skipping training and evaluation.")
        return

    # ── Per-experiment loop ───────────────────────────────────────────────────
    # model.names allows looping over multiple model families in one run (e.g.
    # ["linear", "xgboost", "tabpfn"]). Falls back to the single model.name for
    # backward compatibility. Both must resolve to at least one entry -- unlike
    # the old cfg["model"]["name"] bracket access, this is an explicit check
    # rather than a bare KeyError when neither is set.
    model_family_list: list[str] = validated.model.names or (
        [validated.model.name] if validated.model.name else []
    )
    if not model_family_list:
        raise ValueError("Config must set model.name or model.names")

    # model.type is normally a list (["regression"] or
    # ["regression", "binary_classification"]) to loop over multiple task types
    # in one run; a bare string is also accepted and normalized to a
    # single-element list here -- previously a bare string silently iterated
    # over its characters instead (see documentation/3_adding_a_new_model.md).
    model_task_types: list[str] = (
        [validated.model.type] if isinstance(validated.model.type, str) else validated.model.type
    )

    # Noise levels: noise.sigma may be a scalar or a list.
    noise_levels: list[float] = (
        [float(validated.noise.sigma)] if not isinstance(validated.noise.sigma, list)
        else [float(s) for s in validated.noise.sigma]
    )

    # Random row fractions: data.rand may be a scalar or a list.
    rand_fracs: list[float] = (
        [float(validated.data.rand)] if not isinstance(validated.data.rand, list)
        else [float(r) for r in validated.data.rand]
    )

    # PCA settings: data.pca may be a scalar, list, or null.
    # Accepted values: float (0,1) for variance threshold, "effective", null.
    _raw_pca = validated.data.pca
    if _raw_pca is None:
        _raw_pca = [None]
    elif not isinstance(_raw_pca, list):
        _raw_pca = [_raw_pca]
    def _parse_pca(v):
        if v is None:
            return None
        if str(v).lower() == "effective":
            return "effective"
        return float(v)
    pca_values: list[float | str | None] = [_parse_pca(v) for v in _raw_pca]

    # data.path lets you point straight at a prepared tabular file (CSV/TSV/TXT),
    # bypassing the illness/p_clump/distribution GWAS pipeline entirely. Every
    # other data.* setting keeps its default; the target column is resolved via
    # data.target or data.target_regex. data.dir does the same but for every
    # matching file in a folder, running one separate experiment per file.
    custom_data_path = validated.data.path
    custom_data_dir = validated.data.dir
    using_custom_data = bool(custom_data_path or custom_data_dir)

    # These data.* keys belong to the GWAS illness/phenotype construction
    # pipeline and are meaningless once data.path/data.dir + data.target(_regex)
    # point straight at a prepared tabular file — ignore them entirely rather
    # than letting a stray leftover value from a copy-pasted config silently
    # take effect.
    _GWAS_ONLY_DATA_KEYS = frozenset({
        "illness", "min_density", "density_json", "gwas_pheno_path",
        "info_csv_path", "p_clump", "distribution",
        "min_complete_frac", "max_target_corr",
    })
    if using_custom_data:
        _ignored_keys = sorted(k for k in _GWAS_ONLY_DATA_KEYS if k in data_cfg)
        if _ignored_keys:
            print(f"data.path/data.dir is set — ignoring GWAS-pipeline-only "
                  f"data.* keys: {', '.join(_ignored_keys)}")
        data_cfg = {k: v for k, v in data_cfg.items() if k not in _GWAS_ONLY_DATA_KEYS}

    data_path_by_stem: dict[str, Path] = {}

    if custom_data_dir:
        dir_path = Path(custom_data_dir).expanduser().resolve()
        if not dir_path.is_dir():
            raise FileNotFoundError(f"data.dir not found or not a directory: {dir_path}")
        extensions = tuple(e.lower() for e in data_cfg.get("extensions", [".csv", ".tsv", ".txt"]))
        data_files = sorted(
            f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in extensions
        )
        if not data_files:
            raise FileNotFoundError(f"No files with extensions {extensions} found in {dir_path}")
        for f in data_files:
            if f.stem in data_path_by_stem:
                raise ValueError(
                    f"Multiple files in {dir_path} share the name {f.stem!r} "
                    f"({data_path_by_stem[f.stem].name} vs {f.name}) — rename one to disambiguate."
                )
            data_path_by_stem[f.stem] = f
        print(f"data.dir={dir_path} — found {len(data_files)} files: {', '.join(f.name for f in data_files)}")
        illness_list = list(data_path_by_stem)
        distribution_list = ["custom"]
        p_clump_list = [None]
        row_ratio_list = [1.0]
        col_ratio_list = [1.0]
    elif custom_data_path:
        data_path_by_stem[Path(custom_data_path).stem] = Path(custom_data_path)
        illness_list = list(data_path_by_stem)
        distribution_list = ["custom"]
        p_clump_list = [None]
        row_ratio_list = [1.0]
        col_ratio_list = [1.0]
    else:
        # data.illness left blank/empty → auto-detect the phenotype list. Prefer the
        # density-frontier selection (same features used for clumping) when
        # data.min_density is set; otherwise fall back to every include=1 phenotype
        # from info.csv that's present as a column in the gwas_pheno matrix.
        illness_list = validated.data.illness or []
        if not illness_list:
            min_density = validated.data.min_density
            if min_density is not None:
                density_json = validated.data.density_json
                illness_list = select_dense_features(density_json, float(min_density))
                print(f"data.illness is empty — selected {len(illness_list)} phenotypes "
                      f"at density >= {min_density} from {density_json}: "
                      f"{', '.join(illness_list)}")
            else:
                if not validated.data.gwas_pheno_path or not validated.data.info_csv_path:
                    raise ValueError(
                        "data.illness is empty and data.min_density is unset -- "
                        "data.gwas_pheno_path and data.info_csv_path are required "
                        "to auto-detect the phenotype list from info.csv"
                    )
                illness_list = included_phenotype_columns(
                    validated.data.gwas_pheno_path, validated.data.info_csv_path,
                )
                print(f"data.illness is empty — auto-detected {len(illness_list)} phenotypes "
                      f"from info.csv: {', '.join(illness_list)}")
        distribution_list = validated.data.distribution
        p_clump_list = validated.data.p_clump
        row_ratio_list = validated.data.row_ratio
        col_ratio_list = validated.data.col_ratio

    for dist, p, illness, row_ratio, col_ratio, task_type, model_family, noise_sigma, rand_frac, pca_var in product(
        distribution_list,
        p_clump_list,
        illness_list,
        row_ratio_list,
        col_ratio_list,
        model_task_types,
        model_family_list,
        noise_levels,
        rand_fracs,
        pca_values,
    ):
        model_name = resolve_model_name(model_family, task_type)

        # Per-iteration cfg copy: override model.type, noise.sigma, and
        # data.pca_components so downstream calls see the correct values.
        # model.overrides.<family> holds params specific to one model family
        # (e.g. tabpfn's finetune/learning_rate) so looping over model.names
        # doesn't leak one model's extra params into another's pinned params
        # (see src/cv.py::_pinned_params).
        model_overrides = (validated.model.overrides or {}).get(model_family, {})
        iter_cfg = {
            **cfg,
            "model": {**cfg["model"], **model_overrides, "type": task_type},
            "noise": {**cfg.get("noise", {}), "sigma": noise_sigma},
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
        if using_custom_data:
            print(
                f"\nStarting experiment: data_path={data_path_by_stem[illness]}, "
                f"task_type={task_type}, model={model_name}"
                + (f", noise_sigma={noise_sigma:g}"          if noise_sigma > 0    else "")
                + (f", rand={rand_frac:g}"                   if rand_frac  < 1.0   else "")
                + (f", pca={pca_var}"                        if pca_var is not None else "")
            )
        else:
            print(
                f"\nStarting experiment: illness={illness}, p_clump={p},"
                f" distribution={dist}, task_type={task_type}, model={model_name}"
                + (f", noise_sigma={noise_sigma:g}"          if noise_sigma > 0    else "")
                + (f", rand={rand_frac:g}"                   if rand_frac  < 1.0   else "")
                + (f", pca={pca_var}"                        if pca_var is not None else "")
            )
        # Reflects data.whitening from the config, not the runtime fit result --
        # transform_method is a config value known up front, so the filename
        # can be built before results_dir is created below (whitening itself
        # runs later, after the data is loaded).
        _whitening_cfg_for_name = data_cfg.get("whitening")
        _whitening_enabled_for_name = (
            bool(_whitening_cfg_for_name) if isinstance(_whitening_cfg_for_name, bool)
            else bool((_whitening_cfg_for_name or {}).get("enabled", False))
        )
        whitening_suffix = ""
        if _whitening_enabled_for_name:
            _wcfg_for_name = _whitening_cfg_for_name if isinstance(_whitening_cfg_for_name, dict) else {}
            whitening_suffix = f"_whitening_{_wcfg_for_name.get('transform_method', 'zca')}"

        residual_suffix = "_residual" if data_cfg.get("residual", False) else ""
        if residual_suffix == "_residual":
            print("The residual flag is set — the target y will be replaced with the residual of an out-of-fold linear regression on the same features before training.")

        same_category_suffix = "_samecatexcl" if data_cfg.get("exclude_same_category", False) else ""

        if using_custom_data:
            experiment_name = f"{model_name}_{illness}_{task_type}{noise_suffix}{rand_suffix}{pca_suffix}{whitening_suffix}{residual_suffix}{same_category_suffix}"
        else:
            experiment_name = f"{model_name}_{illness}_p{p}_{dist}_{row_ratio}_{col_ratio}_{task_type}{noise_suffix}{rand_suffix}{pca_suffix}{whitening_suffix}{residual_suffix}{same_category_suffix}"
        results_dir = Path("./results") / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{experiment_name}_{timestamp}.json"
        output = {}

        # ── Load data ─────────────────────────────────────────────────────────
        if using_custom_data:
            resolved_data_path = data_path_by_stem[illness]
            sep = "," if resolved_data_path.suffix.lower() == ".csv" else "\t"
            if data_cfg.get("polars", True):
                df = load_txt_polars(resolved_data_path, sep=sep, chunk_size=chunk_size, total_chunks=total_chunks)
            else:
                df = load_txt(resolved_data_path, sep=sep, chunk_size=chunk_size, total_chunks=total_chunks)
        else:
            # ── Optional sampling ─────────────────────────────────────────────
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
                # than letting the config look like it applied. (whitening is NOT
                # in this list — it runs independently of sampling, below.)
                inert = [k for k in ("max_target_corr", "min_complete_frac", "max_col_missing")
                         if data_cfg.get(k) is not None]
                if inert:
                    print(f"  NOTE: sampling is off — {', '.join(inert)} not applied; "
                          f"reusing {data_path} as written. Set data.sampling: true "
                          f"to recompute (and to report target correlations).")

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
        if not using_custom_data:
            print(f"{row_ratio*100}% of rows, {col_ratio*100}% of columns retained after sampling")
            print(f"Original shape {int(df.shape[0] / row_ratio)} samples, {int(df.shape[1] / col_ratio)} features")

        df_pandas = df.to_pandas() if hasattr(df, "to_pandas") else df

        # ── Optional: drop predictors sharing the target's own trait category ──
        # Runs first — before drop_missing (row-level) and before the target-
        # correlation / whitening feature-pruning steps below — so a same-
        # category predictor is never a candidate feature at all, rather than
        # being trained on and only excluded post-hoc from an analysis. This
        # is the causal counterpart to figures/'s post-hoc same-category-cell
        # exclusion: with the predictor never offered to the model, any
        # power-adjusted gain that survives can't be routed through it.
        same_category_stats = None
        if data_cfg.get("exclude_same_category", False):
            df_pandas, same_category_stats = drop_same_category_features(df_pandas, illness)
            if same_category_stats["target_category"] is None:
                print(f"  exclude_same_category: {illness!r} not found in the GWAS reference "
                      "sheet — no category to exclude by, all features kept")
            else:
                print(f"  exclude_same_category: target category="
                      f"{same_category_stats['target_category']!r}, dropped "
                      f"{same_category_stats['n_features_dropped']} same-category feature(s) "
                      f"({same_category_stats['n_features_kept']}/"
                      f"{same_category_stats['n_features_before']} kept)")
            output[f"same_category_exclusion_{illness}"] = same_category_stats

        # Resolve the target column: an explicit data.target name takes
        # priority; otherwise data.target_regex is matched against the
        # dataframe's columns (must match exactly one).
        target_col = data_cfg.get("target")
        target_regex = data_cfg.get("target_regex")
        if not target_col:
            if not target_regex:
                raise ValueError("Config must set data.target or data.target_regex")
            matches = [c for c in df_pandas.columns if re.search(target_regex, c)]
            if len(matches) != 1:
                raise ValueError(
                    f"data.target_regex={target_regex!r} matched {len(matches)} columns "
                    f"(expected exactly 1): {matches}"
                )
            target_col = matches[0]
            print(f"Resolved target column via regex: {target_col}")
        iter_cfg["data"]["target"] = target_col

        # ── Optional: drop rows with any missing value (complete-case matrix) ──
        # Feature columns (other phenotypes) can be null at a phenotype's selec
        # SNPs; defaults on (a complete-case matrix is the safe default for a
        # generic new dataset) — set data.drop_missing: false to keep rows with
        # missing values.
        # Runs before whitening (moved here from after max_target_corr) so
        # drop_missing_stats reports the true missingness in the loaded data —
        # apply_whitening also drops incomplete rows internally, and with
        # drop_missing running first that inner drop becomes a no-op, keeping
        # the row-count accounting in one place instead of split across two
        # silently-overlapping steps.
        drop_missing_stats = None
        if data_cfg.get("drop_missing", True):
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

        # ── Optional whitening (data.whitening in the YAML) ─────────────────────
        # Decorrelates the target and the surviving features by Sigma (the
        # LDSC/JASS intercept covariance) — removes nuisance cross-trait
        # correlation (sample overlap, population structure) that Sigma
        # captures, before the blunt target-correlation cutoff below gets a
        # chance to act on it. Runs here — independent of data.sampling,
        # exactly once, on whatever matrix is about to be trained on — rather
        # than inside sample(), so it isn't skipped when sampling: false reuses
        # an existing file, and can't be double-applied when sampling: true
        # does write a fresh one. See dataloader/whitening.py.
        whitening_cfg = data_cfg.get("whitening")
        whitening_enabled = (
            bool(whitening_cfg) if isinstance(whitening_cfg, bool)
            else bool((whitening_cfg or {}).get("enabled", False))
        )
        whitening_stats = None
        if whitening_enabled:
            if not data_cfg.get("gwas_pheno_path"):
                raise ValueError(
                    "data.whitening requires data.gwas_pheno_path — Sigma's "
                    "trait labels only match the gwas_pheno phenotype panel's "
                    "column names, not the MRI-phenotype pipeline's."
                )
            from dataloader.whitening import apply_whitening, fit_whitener

            wcfg = whitening_cfg if isinstance(whitening_cfg, dict) else {}
            id_cols_pre = [c for c in ["ID"] if c in df_pandas.columns]
            feature_cols = [c for c in df_pandas.columns
                             if c not in id_cols_pre + [target_col]]
            fit = fit_whitener(
                sigma_path=wcfg.get(
                    "sigma_path", "data/pipeline/input/gwas_pheno/official_intercept_matrix.csv"),
                target=illness, feature_names=feature_cols,
                method=wcfg.get("method", "ridge"),
                n_null=int(wcfg.get("n_null", 5000)),
                tail_ratio_tol=float(wcfg.get("tail_ratio_tol", 2.0)),
                seed=int(wcfg.get("seed", 0)),
                # False -> sigma_path is assumed already PD as a whole (e.g.
                # pre-regularized via `python -m dataloader.whitening --out`);
                # every [target]+features submatrix of a PD matrix is itself
                # PD, so no per-target search is needed. True (default) keeps
                # the null-calibration auto-search for a raw, unregularized Sigma.
                search_alpha=bool(wcfg.get("search_alpha", False)),
                transform_method=wcfg.get("transform_method", "zca")
            )
            df_pandas, whitening_stats = apply_whitening(
                df_pandas, fit, target_col=target_col)
            skipped = whitening_stats["sweep"].get("skipped")
            if skipped == "search_alpha=False -- Sigma assumed already PD":
                header = (f"loaded SPD matrix, Ridge regularization skipped "
                          f"(lambda_min={whitening_stats['lambda_min_raw']:.4g})")
            elif skipped == "already positive definite":
                header = (f"submatrix already positive definite "
                          f"(lambda_min={whitening_stats['lambda_min_raw']:.4g}), "
                          "no regularization needed")
            else:
                header = (f"alpha={whitening_stats['alpha']:.4g} "
                          f"({whitening_stats['method']}), lambda_min "
                          f"{whitening_stats['lambda_min_raw']:.4g} -> "
                          f"{whitening_stats['lambda_min_regularized']:.4g}")
            print(f"  whitening: {header}, "
                  f"{whitening_stats['n_features']} feature(s) whitened "
                  f"({len(whitening_stats['dropped_features'])} not found in Sigma), "
                  f"{whitening_stats['n_rows_dropped_incomplete']} row(s) dropped "
                  "for missing values before whitening")

        # ── Marginal correlation with the target ──────────────────────────────
        # Applied to whatever matrix is about to be trained on, so it also
        # covers sampling: false (sampled files on disk predate this filter).
        # Runs after drop_missing and whitening now, so this is the correlation
        # of the final, complete-case (and possibly whitened) training matrix —
        # not a pairwise-complete estimate on the raw data. When sampling ran,
        # this is a no-op re-check — the report then simply describes the
        # final training matrix.
        target_corr_stats = None
        max_target_corr = data_cfg.get("max_target_corr", None if using_custom_data else 0.8)
        if max_target_corr is not None:
            df_pandas, target_corr_stats = drop_target_correlated_features(
                df_pandas, float(max_target_corr), target=target_col,
            )
            print(f"  target correlation (max_target_corr={max_target_corr}): kept "
                  f"{target_corr_stats['n_features_kept']}/{target_corr_stats['n_features_before']} "
                  f"features ({target_corr_stats['n_features_dropped']} dropped"
                  + (f": {', '.join(target_corr_stats['dropped_columns'])}"
                     if target_corr_stats["dropped_columns"] else "") + ")")

        # data.ignore_columns lists columns (e.g. ID/metadata columns) to drop
        # from the feature matrix in addition to the target column.
        ignore_cols = [c for c in data_cfg.get("ignore_columns", []) if c in df_pandas.columns]
        id_cols = [col for col in ["ID"] if col in df_pandas.columns]
        drop_cols = list(dict.fromkeys([target_col] + id_cols + ignore_cols))
        # Dataset row identifiers, kept alongside X/y (not just dropped with
        # the other id_cols) so predictions and per-row SHAP values saved by
        # nested_cv can be traced back to the sample they came from. Falls
        # back to the positional row index when there's no "ID" column.
        row_ids = (
            df_pandas[id_cols[0]].to_numpy() if id_cols
            else np.arange(len(df_pandas))
        )
        X = df_pandas.drop(columns=drop_cols)
        y = df_pandas[target_col]

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
            row_ids = row_ids[idx]
            print(f"  Random subsample: {n_sample:,}/{n_total:,} rows (rand={rand_frac:g})")

        # ── Optional sign-flip of negative labels + their features ────────────
        if data_cfg.get("invert", True):
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
            # Run HPO; models without a search space fall back to n_trials=0.
            # hpo.n_trials_by_model.<model_family> overrides the shared
            # hpo.n_trials for that one model family, so a job spanning
            # several models (model.names) can give each its own trial budget.
            n_trials = hpo_cfg.get("n_trials_by_model", {}).get(
                model_family, hpo_cfg.get("n_trials", 30)
            )
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
            feature_names=list(X.columns),
            results_dir=results_dir,
            row_ids=row_ids,
        )

        output.update({
            "experiment": experiment_name,
            "noise_sigma": noise_sigma,
            "rand_frac": rand_frac,
            "pca": pca_var,
            "target_correlation": target_corr_stats,
            "drop_missing": drop_missing_stats,
            "whitening": whitening_stats,
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
