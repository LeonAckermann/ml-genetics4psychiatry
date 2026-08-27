"""Experiment config: the Pydantic schema that validates a config YAML end to
end, plus the interactive wizard that writes one.

## Schema (RootConfig and friends)

Validates every top-level block of an experiment config at startup --
catching typos and wrong types before a job runs, rather than letting a
misspelled key be silently ignored by ``dict.get(...)`` somewhere deep in the
pipeline. ``RootConfig`` is the entry point; ``validate_config()`` in main.py
builds one from the raw YAML dict and reports every error at once.

This is validation-only: main.py and src/ still read the original dict via
``cfg.get(...)`` throughout -- migrating that to actually *consume* these
models (so a schema default is the only place a default lives, instead of
also being duplicated in a `.get(key, default)` call) is a separate, later
step.

## Wizard (build_config / main)

    python -m src.config

Walks through data source, model(s), task type, and evaluation mode one
question at a time, skipping anything that doesn't apply to earlier answers.
Every question maps to a field in RootConfig above -- the wizard itself never
invents a default, it just reads the field's schema default and offers it as
the prompt's default. The result is validated against RootConfig before being
written, so a config this wizard produces can never fail main.py's own
validate_config() gate.

When several models are chosen, you're asked whether to write one combined
config (model.names, all models in the same job) or one separate config per
model -- see build_config(). Each written config file gets its own,
independently-configurable batch script (own CPU/GPU choice, device count,
memory, logs dir) -- see _ask_batch_files() -- matching the SBATCH
conventions already used in batch_scripts/ (module load Python, source the
venv, TABPFN_ALLOW_CPU_LARGE_DATASET only when that specific config uses
tabpfn, python3.12 main.py --config ...).

This is a first pass covering the common case: a training-only config (an
already-curated dataset, either a plain file/directory or an existing GWAS
phenotype-genotype matrix). It does not walk through the data-curation
blocks (construct_gwas_mri / plink2.prepare / gwas_phenotype_construction /
phenotype_clumping) -- those are one-off, per-dataset setup better hand-written
from documentation/1_data_curation_pathA.md / 1_data_curation_pathB.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Keys that used to do something but no longer do. Kept as declared (not
# rejected) so old configs still validate; a warning is printed once if any
# of them is actually set, rather than failing the run outright.
_DEPRECATED_DATA_KEYS = ("test_size", "n_splits", "save_splits")
_DEPRECATED_EVALUATION_KEYS = ("metrics", "binary_threshold")
# Leftovers from an earlier SHAP implementation (the plain `shap` library's
# force/waterfall/dependence plots, a permutation explainer) predating the
# switch to shapiq -- src/shap.py never reads any of these.
_DEPRECATED_SHAP_KEYS = (
    "dependence_features", "dependence_dpi", "waterfall_observations",
    "force_stacked", "force_stacked_max_rows", "permutation_max_evals",
    "tabpfn_budget",
)


def _warn(block: str, present: list[str]) -> None:
    if present:
        print(
            f"  NOTE: {block}.{{{', '.join(present)}}} set in config but no "
            "longer read by the pipeline -- safe to remove."
        )


# ---------------------------------------------------------------------------
# data:
# ---------------------------------------------------------------------------

class WhiteningConfig(BaseModel):
    """`data.whitening` when given as a dict (a bare `true`/`false` is also valid)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sigma_path: str = "data/pipeline/input/gwas_pheno/official_intercept_matrix.csv"
    method: str = "ridge"
    n_null: int = 5000
    tail_ratio_tol: float = 2.0
    seed: int = 0
    search_alpha: bool = True
    transform_method: str = "zca"


class DataConfig(BaseModel):
    """Validates the `data:` block of an experiment YAML.

    Two loading paths share this one block:
      - generic:      data.path / data.dir + data.target / data.target_regex
      - specialized:  GWAS phenotype-genotype matrix (data.illness, data.gwas_pheno_path, ...)

    See documentation/2_data_preprocessing_generic.md and
    documentation/2_data_preprocessing_specialized.md for what each field does.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Generic path ──────────────────────────────────────────────────────
    path: str | None = None
    dir: str | None = None
    extensions: list[str] = Field(default_factory=lambda: [".csv", ".tsv", ".txt"])
    target: str | None = None
    target_regex: str | None = None

    # ── Specialized (GWAS) path ──────────────────────────────────────────
    # NOTE: `illness` is a single plain string in the one-time
    # construct_gwas_mri / plink2.prepare curation blocks, but a *list* here
    # in the per-experiment training loop -- see the warning in
    # documentation/1_data_curation_pathA.md. Both shapes are accepted.
    illness: str | list[str] = Field(default_factory=list)
    min_density: float | None = None
    density_json: str = "./data/pipeline/analysis/dense_density.json"
    gwas_pheno_path: str | None = None
    info_csv_path: str | None = None
    p_clump: list[float] = Field(default_factory=list)
    distribution: list[str] = Field(default_factory=list)
    sample_p: bool = False
    clumps_path: str | None = None
    use_parquet: bool = True

    # ── Shared preprocessing knobs (both paths) ──────────────────────────
    drop_missing: bool = True
    polars: bool = True
    row_ratio: list[float] = Field(default_factory=lambda: [1.0])
    col_ratio: list[float] = Field(default_factory=lambda: [1.0])
    top_rows: bool = True
    top_cols: bool = True
    sampling: bool = False
    invert: bool = False
    residual: bool = False
    residual_baseline_model: str = "linear_regression"
    exclude_same_category: bool = False
    ignore_columns: list[str] = Field(default_factory=list)
    rand: float | list[float] = 1.0
    pca: Any = None  # float in (0,1), "effective", null, or a list of those
    chunk_size: int = 100000
    total_chunks: int | None = None
    mri_p_value: float = 0.05
    max_col_missing: float | None = None
    min_complete_frac: float | None = None
    max_target_corr: float | None = 0.8
    whitening: bool | WhiteningConfig = False

    # ── Deprecated -- see _DEPRECATED_DATA_KEYS ──────────────────────────
    test_size: float | None = None
    n_splits: int | None = None
    save_splits: bool | None = None

    @model_validator(mode="after")
    def _warn_deprecated(self) -> "DataConfig":
        _warn("data", [k for k in _DEPRECATED_DATA_KEYS if getattr(self, k) is not None])
        return self


# ---------------------------------------------------------------------------
# plink2: / construct_gwas_mri: / gwas_phenotype_construction: / phenotype_clumping:
# (data-curation stage -- see documentation/1_data_curation*.md)
# ---------------------------------------------------------------------------

class Plink2Config(BaseModel):
    """`plink2:` -- Path A curation, stage 2 (align + LD-clump). Required top-level
    key even when unused (`prepare: false`); see documentation/1_data_curation_pathA.md."""

    model_config = ConfigDict(extra="forbid")

    prepare: bool = False
    p_clump: float = 1
    r2: float = 0.05
    clump_kb: int = 500
    chunk_size: int = 100000
    total_chunks: int | None = None
    mri: str | None = None
    ref: str | None = None
    aligned: str | None = None
    output: str | None = None
    polars: bool = False
    create_final: bool = True
    output_suffix: str = ""

    @model_validator(mode="after")
    def _require_when_prepared(self) -> "Plink2Config":
        if self.prepare:
            missing = [k for k in ("mri", "ref", "aligned", "output") if getattr(self, k) is None]
            if missing:
                raise ValueError(
                    f"plink2.prepare is true but missing required key(s): {', '.join(missing)}"
                )
        return self


class ConstructGwasMriConfig(BaseModel):
    """`construct_gwas_mri:` -- Path A curation, stage 1. See
    documentation/1_data_curation_pathA.md."""

    model_config = ConfigDict(extra="forbid")

    run: bool = False
    input_path: str | None = None
    output_path: str = "data/pipeline/input/gwas_mri/all_z_scores.txt"
    chunk_size: int = 10000
    total_chunks: int | None = None
    polars: bool = False
    value: str = "T_STAT"

    @model_validator(mode="after")
    def _require_when_run(self) -> "ConstructGwasMriConfig":
        if self.run and self.input_path is None:
            raise ValueError("construct_gwas_mri.run is true but input_path is not set")
        return self


class GwasPhenotypeConstructionConfig(BaseModel):
    """`gwas_phenotype_construction:` -- Path B curation, stage 1. See
    documentation/1_data_curation_pathB.md."""

    model_config = ConfigDict(extra="forbid")

    run: bool = False
    input_path: str | None = None
    output_path: str | None = None
    how: Literal["inner", "outer"] = "inner"
    join_key: Literal["chrom", "rs"] = "chrom"
    info_csv_path: str | None = None

    @model_validator(mode="after")
    def _require_when_run(self) -> "GwasPhenotypeConstructionConfig":
        if self.run:
            missing = [k for k in ("input_path", "output_path") if getattr(self, k) is None]
            if missing:
                raise ValueError(
                    f"gwas_phenotype_construction.run is true but missing required "
                    f"key(s): {', '.join(missing)}"
                )
        return self


class PhenotypeClumpingConfig(BaseModel):
    """`phenotype_clumping:` -- Path B curation, stage 2. See
    documentation/1_data_curation_pathB.md."""

    model_config = ConfigDict(extra="forbid")

    run: bool = False
    gwas_pheno_path: str | None = None
    ref: str | None = None
    info_csv_path: str | None = None
    phenotypes: list[str] | None = None
    min_density: float | None = None
    density_json: str = "./data/pipeline/analysis/dense_density.json"
    p_clump: float = 1
    r2: float = 0.05
    clump_kb: int = 500
    create_final: bool = False

    @model_validator(mode="after")
    def _require_when_run(self) -> "PhenotypeClumpingConfig":
        if self.run:
            missing = [k for k in ("gwas_pheno_path", "ref") if getattr(self, k) is None]
            if missing:
                raise ValueError(
                    f"phenotype_clumping.run is true but missing required key(s): "
                    f"{', '.join(missing)}"
                )
        return self


# ---------------------------------------------------------------------------
# hpo:
# ---------------------------------------------------------------------------

class CVConfig(BaseModel):
    """`cv:` -- selects how the outer cross-validation folds are built.

    `type: chromosome` groups rows (SNPs) by chromosome -- looked up by each
    row's `ID` (rsID) in `chrom_map_path`, the joined GWAS z-score matrix that
    still carries `chrom` (the trained-on feature matrix itself drops it) --
    and evaluates leave-one-chromosome-out, one fold per chromosome present
    (22 for the standard autosomes). This mode always disables hyperparameter
    search (see main.py): every fold trains with the plain `model:` config
    values, since there's no natural inner split to nest a search inside a
    single held-out chromosome. `hpo.outer_cv` is ignored under this mode --
    the fold count is however many chromosomes are actually present.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["kfold", "chromosome"] = "kfold"
    chrom_map_path: str = "data/pipeline/input/gwas_pheno/all_z_scores_imputed.txt"


class HPOConfig(BaseModel):
    """Validates the `hpo:` block when it's a dict (a bare `hpo: true` is also valid
    and bypasses this schema -- see main.py's hpo_enabled resolution)."""

    model_config = ConfigDict(extra="forbid")

    run: bool = True
    n_trials: int = 30
    inner_cv: int = 5
    outer_cv: int = 5
    search_space: dict[str, Any] | None = None
    # Per-model override of n_trials, keyed by model.name/model.names entry --
    # falls back to n_trials above for any model not listed here. Only
    # meaningful for models with a default search space to begin with (see
    # documentation/3_model_hyperparameters.md); n_trials is auto-set to 0
    # regardless of either value for a model with none.
    n_trials_by_model: dict[str, int] | None = None
    # Which Optuna sampler explores the search space -- see
    # documentation/3_nested_cross_fold_validation.md#hyperparameter-optimization.
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"


# ---------------------------------------------------------------------------
# model:
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Validates the `model:` block.

    Unlike the other blocks, this one allows arbitrary extra scalar keys --
    any scalar (non-list, non-dict) value here is passed straight through to
    ``build_model()`` as a "pinned" hyperparameter (see
    ``src/cv.py::_pinned_params`` and documentation/3_nested_cross_fold_validation.md),
    so it's an intentionally open bag, not a fixed field list. The well-known
    routing/extras fields below are still validated by type.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    names: list[str] | None = None
    # KNOWN BUG (not yet fixed -- see the migration step): main.py's product()
    # call currently reads `model.type` expecting it to already be a list;
    # a bare string like "regression" is iterated character-by-character.
    # Always write `type: [regression]`, not `type: regression`, until fixed.
    type: str | list[str] = "regression"
    types: list[str] | None = None  # currently dead -- computed but unused in main.py
    overrides: dict[str, dict[str, Any]] | None = None
    device: str = "cpu"
    p_value_binary: float | None = None  # required (crashes without it) if type includes binary_classification
    finetune: bool | None = None
    learning_rate: float | None = None
    epochs: int | None = None


# ---------------------------------------------------------------------------
# experiment: / noise: / evaluation: / shap: / predictions: / training_curves:
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: bool = True


class NoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sigma: float | list[float] = 0.0


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = 0.5  # MDN confidence-metric reporting threshold only

    # ── Deprecated -- see _DEPRECATED_EVALUATION_KEYS ────────────────────
    metrics: list[str] | None = None
    binary_threshold: float | None = None

    @model_validator(mode="after")
    def _warn_deprecated(self) -> "EvaluationConfig":
        _warn("evaluation", [k for k in _DEPRECATED_EVALUATION_KEYS if getattr(self, k) is not None])
        return self


class ShapConfig(BaseModel):
    """`shap:` -- see documentation/5 (feature attribution) and src/shap.py."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interactions: bool = False
    index: str | None = None  # default: "k-SII" if interactions else "SV"
    max_order: int | None = None  # default: 2 if interactions else 1
    budget: str | int = "auto"
    max_budget: int = 2048
    background_size: int = 100
    random_state: int = 0
    max_explain_rows: int | None = None  # None -> explain every row
    store: bool = False  # persist per-row values, not just the fold/experiment mean
    n_jobs: int | None = None
    verbose: bool = False
    plots: bool = True
    beeswarm_max_rows: int = 5000
    abbreviate_names: bool = False
    max_display: int = 20
    imputer_sample_size: int | None = None  # default: len(background) if unset

    # ── Deprecated -- see _DEPRECATED_SHAP_KEYS ──────────────────────────
    dependence_features: list[str] | None = None
    dependence_dpi: int | None = None
    waterfall_observations: list[Any] | None = None
    force_stacked: bool | None = None
    force_stacked_max_rows: int | None = None
    permutation_max_evals: int | None = None
    tabpfn_budget: int | None = None

    @model_validator(mode="after")
    def _warn_deprecated(self) -> "ShapConfig":
        _warn("shap", [k for k in _DEPRECATED_SHAP_KEYS if getattr(self, k) is not None])
        return self


class PredictionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class TrainingCurvesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    plots: bool = False
    dir: str = "training"


class WandbConfig(BaseModel):
    """`wandb:` -- uploads training curves (see `training_curves:`) and final
    metrics to Weights & Biases, one run per experiment (every outer fold's
    curve on the same chart). Requires local W&B auth already set up (env var
    `WANDB_API_KEY` or a prior `wandb login`) -- this config never carries a
    credential. See documentation/4_wandb_logging.md."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "ml-genetics4psychiatry"
    entity: str | None = None  # W&B team/username; None = account default
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: list[str] | None = None


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class RootConfig(BaseModel):
    """Validates an entire experiment YAML file.

    `experiment` and `model` are required -- main.py reads them with
    `cfg["..."]` (no default), so a config missing one already fails today;
    this just fails with a clear message instead of a bare KeyError. `plink2`
    is optional (defaults to `Plink2Config()`, i.e. `prepare: false`) since a
    training-only config has no reason to declare an empty `plink2: {}` block.
    """

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    experiment: ExperimentConfig
    model: ModelConfig

    plink2: Plink2Config = Field(default_factory=Plink2Config)
    construct_gwas_mri: ConstructGwasMriConfig = Field(default_factory=ConstructGwasMriConfig)
    gwas_phenotype_construction: GwasPhenotypeConstructionConfig = Field(
        default_factory=GwasPhenotypeConstructionConfig)
    phenotype_clumping: PhenotypeClumpingConfig = Field(default_factory=PhenotypeClumpingConfig)

    # `hpo: true` (bare bool, meaning "on, with every default") is also valid --
    # see main.py's hpo_enabled resolution -- so this isn't just HPOConfig.
    hpo: bool | HPOConfig = Field(default_factory=lambda: HPOConfig(run=False))
    cv: CVConfig = Field(default_factory=CVConfig)

    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    shap: ShapConfig = Field(default_factory=ShapConfig)
    predictions: PredictionsConfig = Field(default_factory=PredictionsConfig)
    training_curves: TrainingCurvesConfig = Field(default_factory=TrainingCurvesConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    load_best_params: bool = False
    seed: int = 42
    verbose: bool = True

    # ── Deprecated top-level blocks -- never read by main.py ─────────────
    # A top-level `sampling:` block (distinct from `data.sampling`, a bool)
    # is a leftover from an older pipeline shape; ignored today.
    sampling: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _warn_deprecated(self) -> "RootConfig":
        _warn("<root>", ["sampling"] if self.sampling is not None else [])
        return self


# =============================================================================
# Wizard: python -m src.config
# =============================================================================


_AVAILABLE_MODELS = ["linear", "lasso", "ridge", "xgboost", "residual_dnn", "tabpfn"]

# linear and tabpfn have no default HPO search space (src/hpo.py::_DEFAULT_SPACES)
# -- n_trials is auto-set to 0 for them regardless of what's asked, so the
# wizard doesn't bother asking about trial counts for either.
_NO_HPO_SEARCH_SPACE_MODELS = {"linear", "tabpfn"}

_SAMPLERS = ["tpe", "random", "cmaes"]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _ask(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{question}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print("  This is required.")


def _ask_bool(question: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def _ask_float(question: str, default: float) -> float:
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def _ask_int(question: str, default: int) -> int:
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a whole number.")


def _ask_list(question: str, default: list[str] | None = None) -> list[str]:
    # Unlike _ask(), blank is always a valid answer here -- an empty list is
    # a legitimate default (e.g. "blank = auto-detect"), not "missing".
    default = default or []
    shown = ", ".join(default) if default else "blank"
    raw = input(f"{question} (comma-separated) [{shown}]: ").strip()
    if not raw:
        return list(default)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _ask_choice(question: str, options: list[str], default_index: int = 0) -> str:
    print(f"{question}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if i - 1 == default_index else ""
        print(f"  {i}) {opt}{marker}")
    while True:
        raw = input(f"Choice [1-{len(options)}]: ").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number from 1 to {len(options)}.")


def _ask_multi_choice(question: str, options: list[str]) -> list[str]:
    print(f"{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        raw = input(f"Choice(s), comma-separated [1-{len(options)}]: ").strip()
        if not raw:
            print("  Pick at least one.")
            continue
        try:
            idxs = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print("  Please enter number(s) separated by commas.")
            continue
        if all(1 <= i <= len(options) for i in idxs):
            return [options[i - 1] for i in idxs]
        print(f"  Please enter numbers from 1 to {len(options)}.")


# ---------------------------------------------------------------------------
# Wizard sections
# ---------------------------------------------------------------------------

def _ask_data() -> dict[str, Any]:
    print("\n--- Data ---")
    source = _ask_choice(
        "Where is your data?",
        [
            "A single prepared file (data.path)",
            "A directory of prepared files, one experiment per file (data.dir)",
            "An existing GWAS phenotype-genotype matrix (specialized path)",
        ],
    )
    data: dict[str, Any] = {}

    if source.startswith("A single"):
        data["path"] = _ask("Path to the data file (csv/tsv/txt)")
        if _ask_bool("Do you know the exact target column name?", True):
            data["target"] = _ask("Target column name")
        else:
            data["target_regex"] = _ask("Target column regex (must match exactly one column)")

    elif source.startswith("A directory"):
        data["dir"] = _ask("Path to the directory of data files")
        if _ask_bool("Do you know the exact target column name?", True):
            data["target"] = _ask("Target column name")
        else:
            data["target_regex"] = _ask("Target column regex (must match exactly one column)")

    else:
        data["gwas_pheno_path"] = _ask("Path to the joined GWAS phenotype matrix (gwas_pheno_path)")
        data["info_csv_path"] = _ask("Path to the phenotype info.csv (info_csv_path)")
        illness = _ask_list("Phenotype(s)/illness(es) to run (blank = auto-detect)", default=[])
        if illness:
            data["illness"] = illness
        else:
            if _ask_bool("Auto-select by feature density (data.min_density) instead of info.csv?", False):
                data["min_density"] = _ask_float("Minimum feature density (0-1)", 0.8)
        data["p_clump"] = [
            float(v) for v in _ask_list("LD-clumping p-value threshold(s) (data.p_clump)", default=["1"])
        ]
        data["distribution"] = _ask_list("SNP sampling distribution(s) (data.distribution)", default=["low"])
        data["target"] = _ask("Target column name", "Z")

    if not _ask_bool("Drop rows with any missing value (data.drop_missing)?", True):
        data["drop_missing"] = False

    return data


def _ask_model() -> tuple[list[str], dict[str, Any]]:
    """Returns (chosen model families, TabPFN fine-tune overrides -- {} if none)."""
    print("\n--- Model ---")
    chosen = _ask_multi_choice("Which model(s)?", _AVAILABLE_MODELS)

    tabpfn_overrides: dict[str, Any] = {}
    if "tabpfn" in chosen and _ask_bool("Fine-tune TabPFN (model.finetune)?", False):
        tabpfn_overrides = {
            "finetune": True,
            "learning_rate": _ask_float("TabPFN fine-tuning learning rate", 1e-5),
            "epochs": _ask_int("TabPFN fine-tuning epochs", 30),
        }

    return chosen, tabpfn_overrides


def _ask_model_split() -> bool:
    """Only called when more than one model was chosen. True = one config file
    per model; False = one combined config file (model.names)."""
    print("\n--- Multiple models ---")
    choice = _ask_choice(
        "Multiple models selected -- generate one combined experiment file, "
        "or one file per model?",
        [
            "One combined file (model.names -- all models run in the same job)",
            "One separate file per model",
        ],
    )
    return choice.startswith("One separate")


def _ask_task_type() -> dict[str, Any]:
    print("\n--- Task type ---")
    choice = _ask_choice("Task type", ["regression", "binary_classification", "both"])
    task: dict[str, Any] = {}
    if choice == "both":
        task["type"] = ["regression", "binary_classification"]
    else:
        task["type"] = [choice]
    if "binary_classification" in task["type"]:
        task["p_value_binary"] = _ask_float(
            "p-value threshold for the binary label (model.p_value_binary)", 0.05
        )
    return task


def _ask_hpo(models: list[str]) -> dict[str, Any]:
    print("\n--- Evaluation mode ---")
    print("See documentation/3_cross_fold_validation.md / 3_nested_cross_fold_validation.md.")
    mode = _ask_choice(
        "Evaluation mode",
        [
            "Plain cross-fold validation (no hyperparameter search)",
            "Nested cross-fold validation (inner-fold hyperparameter search)",
        ],
    )
    if mode.startswith("Plain"):
        return {"run": False}

    hpo: dict[str, Any] = {"run": True}

    tunable_models = [m for m in models if m not in _NO_HPO_SEARCH_SPACE_MODELS]
    if not tunable_models:
        print("  (none of the chosen models have a hyperparameter search space -- "
              "nested CV will behave like plain CV for all of them; see "
              "documentation/3_model_hyperparameters.md)")
    elif len(tunable_models) > 1 and not _ask_bool(
        "Use the same number of Optuna trials for every model?", True
    ):
        hpo["n_trials_by_model"] = {
            m: _ask_int(f"  Trials for {m} (hpo.n_trials_by_model.{m})", 30) for m in tunable_models
        }
    else:
        hpo["n_trials"] = _ask_int("Optuna trials per outer fold (hpo.n_trials)", 30)

    if tunable_models:
        hpo["sampler"] = _ask_choice("Optuna sampler (hpo.sampler)", _SAMPLERS)

    hpo["outer_cv"] = _ask_int("Outer CV folds (hpo.outer_cv)", 5)
    hpo["inner_cv"] = _ask_int("Inner CV folds per trial (hpo.inner_cv)", 5)

    return hpo


def _ask_extras() -> dict[str, Any]:
    print("\n--- Optional extras ---")
    extras: dict[str, Any] = {}
    if _ask_bool("Enable SHAP feature attribution (shap.enabled)?", False):
        extras["shap"] = {
            "enabled": True,
            "interactions": _ask_bool("  Pairwise interactions (k-SII) instead of plain Shapley values?", False),
        }
    if _ask_bool("Save out-of-fold predictions (predictions.enabled)?", False):
        extras["predictions"] = {"enabled": True}
    return extras


def _ask_logging() -> dict[str, Any]:
    print("\n--- Logging ---")
    print("Training curves (per-fold train/test loss) are saved to the result JSON by default.")
    logging_cfg: dict[str, Any] = {}

    if _ask_bool("Also render training-curve plots locally (training_curves.plots)?", False):
        logging_cfg["training_curves"] = {"plots": True}

    if _ask_bool("Log training curves + final metrics to Weights & Biases (wandb.enabled)?", False):
        print("  Needs W&B auth already set up on this machine (`wandb login`, or a "
              "WANDB_API_KEY env var) -- this wizard never asks for or stores a credential.")
        wandb_cfg: dict[str, Any] = {
            "enabled": True,
            "project": _ask("  W&B project name", "ml-genetics4psychiatry"),
        }
        entity = _ask("  W&B entity (team/username, blank = your account default)", "")
        if entity:
            wandb_cfg["entity"] = entity
        if not _ask_bool("  Sync to the cloud live (no = write offline, `wandb sync` later)?", True):
            wandb_cfg["mode"] = "offline"
        tags = _ask_list("  Tags to attach to the run", default=[])
        if tags:
            wandb_cfg["tags"] = tags
        logging_cfg["wandb"] = wandb_cfg

    return logging_cfg


def _ask_batch_settings(prefix: str = "") -> dict[str, Any]:
    """Ask the CPU/GPU/devices/memory/logs-dir questions for one batch script."""
    batch: dict[str, Any] = {
        "logs_dir": _ask(f"{prefix}Directory for SLURM logs", "logs"),
    }
    device = _ask_choice(f"{prefix}Run on CPU or GPU?", ["CPU", "GPU"])
    batch["gpu"] = device == "GPU"
    batch["n_devices"] = _ask_int(
        f"{prefix}How many GPUs?" if batch["gpu"] else f"{prefix}How many CPUs?",
        1 if batch["gpu"] else 8,
    )
    batch["mem_mb"] = _ask_int(f"{prefix}Memory in MB", 100000)
    return batch


def _ask_batch_files(file_names: list[str]) -> dict[str, dict[str, Any]]:
    """One batch script per written config file (`file_names`, e.g.
    ["multi_test_xgboost", "multi_test_tabpfn"]). Each script is independently
    configurable -- e.g. TabPFN on a GPU while the rest stay on CPU -- since a
    multi-model split is usually split *because* the models have different
    resource needs. Returns {file_name: batch_settings}; a file_name missing
    from the result means "no batch script for this one". Empty dict (not
    None) if batch scripts are declined entirely.
    """
    print("\n--- Batch (SLURM) script(s) ---")
    single = len(file_names) == 1
    verb = "a batch script" if single else f"batch scripts for these {len(file_names)} configs"
    if not _ask_bool(f"Generate {verb}?", False):
        return {}

    if single:
        return {file_names[0]: _ask_batch_settings()}

    if _ask_bool("Use the same batch settings for all of them?", True):
        settings = _ask_batch_settings()
        return {name: settings for name in file_names}

    result: dict[str, dict[str, Any]] = {}
    for name in file_names:
        print(f"\n  -- {name} --")
        if not _ask_bool(f"  Generate a batch script for {name}?", True):
            continue
        result[name] = _ask_batch_settings("  ")
    return result


# ---------------------------------------------------------------------------
# Batch script rendering
# ---------------------------------------------------------------------------

def render_batch_script(
    job_name: str,
    config_paths: list[str],
    batch: dict[str, Any],
    uses_tabpfn: bool,
) -> str:
    """Build a SLURM batch script matching the conventions already used in
    batch_scripts/ (module load Python, source the venv, python3.12 main.py
    --config ...). `config_paths` is usually a single path -- one script per
    config file, each independently configurable (see _ask_batch_files) --
    but takes a list so a caller can still bundle several configs into one
    sequential job if that's ever wanted.
    """
    lines: list[str] = ["#!/usr/bin/env bash", ""]

    lines.append(f"#SBATCH --mem={batch['mem_mb']}")
    lines.append(f"#SBATCH -J {job_name}")
    lines.append(f"#SBATCH -o ./{batch['logs_dir']}/{job_name}_%A_%a.out")
    lines.append(f"#SBATCH -e ./{batch['logs_dir']}/{job_name}_%A_%a.err")
    if batch["gpu"]:
        n = batch["n_devices"]
        lines.append("#SBATCH -p gpu")
        lines.append(f"#SBATCH --gres=gpu:A100{'' if n <= 1 else f':{n}'}")
    else:
        lines.append(f"#SBATCH --cpus-per-task={batch['n_devices']}")
    lines.append("")

    lines.append("module load Python/3.12.9")
    lines.append("")
    lines.append("source ./pasteur/bin/activate")
    if uses_tabpfn:
        lines.append("export TABPFN_ALLOW_CPU_LARGE_DATASET=1")
    lines.append("")

    lines.append('echo "Current working directory: $(pwd)"')
    lines.append("")
    for cfg_path in config_paths:
        lines.append(f'echo "Starting {job_name} ({cfg_path})..."')
        lines.append(f"python3.12 main.py --config {cfg_path}")
        lines.append("")

    lines.append(f'echo "{job_name} completed successfully."')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_config() -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """Returns (configs, chosen_models).

    `configs` is a list of (suffix, cfg) pairs -- exactly one entry with
    suffix="" for a single model or a combined multi-model file; one entry
    per model (suffix = model family name) when the multi-model split was
    chosen.
    """
    print("=== Experiment config wizard ===")
    print("Answers map directly to src/config.py's RootConfig -- press Enter to accept a default.\n")

    data = _ask_data()
    chosen_models, tabpfn_overrides = _ask_model()
    task = _ask_task_type()
    hpo = _ask_hpo(chosen_models)
    extras = _ask_extras()
    logging_cfg = _ask_logging()

    split = len(chosen_models) > 1 and _ask_model_split()

    model_blocks: list[tuple[str, dict[str, Any]]] = []
    if not split:
        if len(chosen_models) > 1:
            model: dict[str, Any] = {"names": chosen_models, **task}
            if tabpfn_overrides:
                model["overrides"] = {"tabpfn": tabpfn_overrides}
        else:
            model = {"name": chosen_models[0], **task}
            if tabpfn_overrides:
                model.update(tabpfn_overrides)
        model_blocks.append(("", model))
    else:
        for m in chosen_models:
            model = {"name": m, **task}
            if m == "tabpfn" and tabpfn_overrides:
                model.update(tabpfn_overrides)
            model_blocks.append((m, model))

    configs: list[tuple[str, dict[str, Any]]] = []
    for suffix, model in model_blocks:
        # Resolve this file's own n_trials_by_model entry (if any) into a
        # plain n_trials, rather than shipping every other model's trial
        # count in a file that only ever trains one of them.
        file_hpo = dict(hpo)
        by_model = file_hpo.pop("n_trials_by_model", None)
        if suffix and by_model and suffix in by_model:
            file_hpo["n_trials"] = by_model[suffix]
        elif by_model and not suffix:
            file_hpo["n_trials_by_model"] = by_model

        cfg: dict[str, Any] = {
            "data": data,
            "model": model,
            "hpo": file_hpo,
            "experiment": {"run": True},
            **extras,
            **logging_cfg,
        }
        configs.append((suffix, cfg))

    return configs, chosen_models


def _cfg_uses_tabpfn(cfg: dict[str, Any]) -> bool:
    model = cfg.get("model", {}) or {}
    return model.get("name") == "tabpfn" or "tabpfn" in (model.get("names") or [])


def main() -> None:
    configs, _chosen_models = build_config()

    for suffix, cfg in configs:
        try:
            RootConfig(**cfg)
        except ValidationError as e:
            label = suffix or "combined"
            print(f"\nSomething the wizard produced doesn't validate ({label}) -- this is a "
                  f"wizard bug, please report it. Details:\n{e}")
            sys.exit(1)

    print("\n--- Save ---")
    base_name = _ask("Experiment name (used for the output filename)")
    out_dir = Path(_ask("Output directory", "experiments"))

    # (file name, path, cfg) for every config actually written to disk.
    written: list[tuple[str, Path, dict[str, Any]]] = []
    for suffix, cfg in configs:
        name = f"{base_name}_{suffix}" if suffix else base_name
        out_path = out_dir / f"{name}.yaml"
        if out_path.exists() and not _ask_bool(f"{out_path} already exists -- overwrite?", False):
            print(f"  Skipped {out_path}.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        header = f"# Experiment: {name}\n# Generated by `python -m src.config` -- edit freely.\n\n"
        with open(out_path, "w") as fh:
            fh.write(header)
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        print(f"Wrote {out_path}")
        written.append((name, out_path, cfg))

    if not written:
        print("\nNothing written.")
        return

    # One independently-configurable batch script per written config file --
    # a multi-model split is usually split *because* the models have
    # different resource needs (e.g. TabPFN on a GPU, everything else on CPU).
    batch_settings = _ask_batch_files([name for name, _, _ in written])

    batch_dir = Path("batch_scripts")
    run_commands: list[str] = []
    for name, out_path, cfg in written:
        settings = batch_settings.get(name)
        if settings is None:
            run_commands.append(f"python main.py --config {out_path}")
            continue

        script = render_batch_script(name, [str(out_path)], settings, _cfg_uses_tabpfn(cfg))
        batch_path = batch_dir / f"{name}.sh"
        if batch_path.exists() and not _ask_bool(f"{batch_path} already exists -- overwrite?", False):
            print(f"  Batch script for {name} not written.")
            run_commands.append(f"python main.py --config {out_path}")
            continue
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(script)
        print(f"Wrote {batch_path}")
        run_commands.append(f"sbatch {batch_path}")

    print("\nRun it with:\n")
    for cmd in run_commands:
        print(f"    {cmd}")
    print()


if __name__ == "__main__":
    main()
