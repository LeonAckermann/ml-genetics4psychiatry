# ML Genetics for Psychiatry — Pipeline Documentation

Documentation for the machine learning pipeline built during an internship at Institut Pasteur.

This pipeline tests machine learning models on the task **f(x) = y**, where `x` is a vector of numerical features and `y` is a continuous or binary variable. This document explains the pipeline's features and nuances so it can be used to its full extent.

```
┌───────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ 1. Data       │──▶│ 2. Data          │──▶│ 3. Model          │──▶│ 4. Performance    │──▶│ 5. Feature           │
│    curation   │   │    preprocessing │   │    optimization   │   │    report         │   │    attribution        │
│               │   │                  │   │                   │   │                   │   │                       │
│ raw GWAS  →   │   │ prepared file →  │   │ nested CV + HPO,  │   │ metrics per fold, │   │ SHAP / shapiq values, │
│ phenotype-    │   │ clean X, y       │   │ one or many       │   │ aggregated,       │   │ per fold, aggregated  │
│ genotype      │   │ matrix           │   │ models per job    │   │ saved locally     │   │ across folds          │
│ matrix        │   │                  │   │                   │   │                   │   │                       │
└───────────────┘   └──────────────────┘   └───────────────────┘   └───────────────────┘   └──────────────────────┘
```

Run any experiment with:

```bash
python main.py --config experiments/your_experiment.yaml
```

## Table of contents

1. [Data curation](#1-data-curation)
2. [Data preprocessing](#2-data-preprocessing)
3. [Model optimization](#3-model-optimization)
4. [Performance report](#4-performance-report)
5. [Feature attribution](#5-feature-attribution)

---

## 1. Data curation

Turns raw per-phenotype GWAS summary-statistic files into one SNP × phenotype matrix. One-time work per illness/phenotype panel — **skip this section if the dataset already exists**.

Two curation paths exist, both orchestrated from [`main.py`](main.py) and implemented in [`dataloader/pipeline.py`](dataloader/pipeline.py):

```
raw per-illness
GWAS files
   │
   ├──▶ Path A — single illness
   │      construct_gwas_mri() ─▶ align + plink2 clump ─▶ aligned_clumped_{ILLNESS}.txt
   │
   └──▶ Path B — multi-phenotype panel
          construct_gwas_phenotype() ─▶ per-phenotype plink2 clump ─▶ clumped_{phenotype}.clumps
```

| | Path A | Path B |
|---|---|---|
| Scope | One illness at a time | Many phenotypes in one joined matrix |
| Config keys | `construct_gwas_mri`, `plink2` | `gwas_phenotype_construction`, `phenotype_clumping` |

→ **[documentation/1_data_curation.md](documentation/1_data_curation.md)** for the full diagrams and complete config examples, or straight to the per-path detail: **[Path A](documentation/1_data_curation_pathA.md)** / **[Path B](documentation/1_data_curation_pathB.md)**.

---

## 2. Data preprocessing

Loads whatever the curation stage (or an already-existing dataset) produced, and turns it into a clean `X`/`y` matrix ready for training. There are two versions:

```
                              ┌─ data.path / data.dir set? ──yes──▶ generic path
data.* config  ───────────────┤
                              └─ no ──▶ specialized GWAS phenotype-panel path
```

### 2a. Generic — `data.path` / `data.dir` + `data.target` / `data.target_regex`

The simple case: point the pipeline straight at one prepared tabular file, or a directory of them (one experiment per file), and name the target column (exactly, or by regex). Every GWAS-specific knob (`illness`, `p_clump`, `distribution`, `gwas_pheno_path`, `whitening`, …) is automatically ignored on this path, and every other setting defaults to something sensible — this is the entire required config:

```yaml
data:
  path: "data/my_dataset.csv"   # a single file ...
  # dir: "data/my_datasets/"    # ... or a directory — one experiment per matching file
  target: "y"                   # exact target column name ...
  # target_regex: "^Z_scores_"  # ... or a regex matched against columns (must match exactly 1)
```

→ **[documentation/2_data_preprocessing_generic.md](documentation/2_data_preprocessing_generic.md)** for the full flow diagram, every default value, and a caveat about the binary-classification target conversion (it assumes a Z-score-shaped target — see that doc before using `binary_classification` on a non-GWAS target).

### 2b. Specialized — GWAS phenotype-genotype matrix

The full-featured path, used whenever `data.path`/`data.dir` are *not* set: phenotype/illness auto-detection, on-the-fly SNP sampling (with feature-density and target-correlation pruning), significance-based row/column filtering, same-trait-category feature exclusion, Σ-based whitening, and an out-of-fold linear-regression residual target.

→ **[documentation/2_data_preprocessing_specialized.md](documentation/2_data_preprocessing_specialized.md)** for the full flow diagram, every config key, and a full example `data:` section.

---

## 3. Model optimization

One generic dispatch point, [`build_model()`](src/hpo.py) in `src/hpo.py`, instantiates every model from a `model_name` + params dict. Every model goes through the same choice: evaluate with plain cross-fold validation, or with nested cross-fold validation that runs its own hyperparameter search inside each outer fold.

```
                                model.name(s) selected
                                          │
                                          ▼
                              evaluation mode chosen per model
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                              ▼
        cross-fold validation                     nested cross-fold validation
             (n_trials = 0)                             (hpo.run: true)
                   │                                              │
                   ▼                                              ▼
      train once per outer fold with          each outer fold: inner-fold HPO
      config-file hyperparameters              search (Optuna) → train + evaluate
                   │                            with that fold's best hyperparameters
                   └──────────────────────┬──────────────────────┘
                                          ▼
                        aggregated metrics across outer folds
```

| Step | What happens | Config |
|---|---|---|
| 1. Select model(s) | Choose which model(s) to run in this job | `model.name` (single) or `model.names` (list) |
| 2. Choose evaluation mode | Per model: plain cross-fold validation, or nested cross-fold validation with hyperparameter optimization | `hpo.run: false` (or no `hpo:` block) → plain CV; `hpo.run: true` → nested CV + HPO |
| 3a. Cross-fold validation | Train once per outer fold, using the config-file (or default) hyperparameters | `n_trials = 0` (automatic when HPO is off, or when a model has no default search space) |
| 3b. Nested cross-fold validation | Each outer fold runs its own inner-fold Optuna search first, then trains + evaluates with the best hyperparameters found for that fold | `hpo.n_trials`, `hpo.inner_cv` |
| 4. Aggregate | Per-fold metrics are averaged (mean/std) across outer folds | → [4. Performance report](#4-performance-report) |

### Available models

| Model | `model.name` | Status |
|---|---|---|
| Linear Regression | `linear` | implemented |
| Lasso | `lasso` | implemented |
| Ridge | `ridge` | implemented |
| XGBoost | `xgboost` | implemented |
| Residual DNN | `residual_dnn` | implemented |
| TabPFNv3 | `tabpfn` | implemented |

> `TabPFNv3` is the model behind the `tabpfn` config name (the shipped checkpoint is `model/tabpfn-v3-regressor-v3_*.ckpt`). This table is the primary, curated model set — a couple of extra models still exist in the code (e.g. `dnn`, `mdn`) but aren't part of the current supported set; see [documentation/3_adding_a_new_model.md](documentation/3_adding_a_new_model.md#available-models) for the full, current list.

### 3.1 Cross-fold validation

Each fold trains once, with no inner hyperparameter search — the simple, cheap default for models with few or no real hyperparameters.

→ **[documentation/3_cross_fold_validation.md](documentation/3_cross_fold_validation.md)** — diagram, config example, and when to use it.

### 3.2 Nested cross-fold validation

Each outer fold runs its own inner-fold hyperparameter search before being scored, so reported metrics are never inflated by hyperparameters chosen using the same data they're evaluated on.

→ **[documentation/3_nested_cross_fold_validation.md](documentation/3_nested_cross_fold_validation.md)** — diagram, config example, and how the inner-fold Optuna search and parameter pinning work (§3.2.1).

→ **[documentation/3_model_hyperparameters.md](documentation/3_model_hyperparameters.md)** — the tunable hyperparameters and default search range for every model above (§3.2.2).

### Running multiple models in one job

`model.names` (list) loops over model families in the same job, alongside every other axis (task type, phenotype, noise level, …):

```yaml
model:
  names: [linear, xgboost, tabpfn]
  overrides:
    tabpfn:
      finetune: true
      learning_rate: 1.0e-5
```

`model.overrides.<name>` scopes model-specific extra parameters (like TabPFN's `finetune`/`learning_rate`) to that model only, so they don't leak into another model's pinned hyperparameters when they share a key name (e.g. `learning_rate` also being a DNN hyperparameter). Falls back to the single `model.name` for backward compatibility.

### Adding a new model

→ **[documentation/3_adding_a_new_model.md](documentation/3_adding_a_new_model.md)** — checklist for implementing a model, wiring it into `build_model()`, and registering its scaling/val-split/HPO-search-space needs.

---

## 4. Performance report

### Metrics

Computed per outer fold via [`src/evaluation.py`](src/evaluation.py), then aggregated (mean/std) across folds by `aggregate_metrics`:

- **Regression**: `r2`, `pearson_r`/`pearson_r2` (+ p-value), `spearman_rho`/`spearman_rho2` (+ p-value).
- **Binary classification**: `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `roc_auc`, `pearson_r2`, confusion-matrix counts (`tn`/`fp`/`fn`/`tp`).

### Weights & Biases

Optional, off by default (`wandb.enabled: false`). When on, [`src/log.py`](src/log.py) uploads one run per experiment, logged **live as training happens** — every epoch (or, for XGBoost, every boosting round) is pushed to W&B the instant it's computed, not batched after the fact. Each fold's selected hyperparameters (from HPO or pinned config) are logged the moment they're resolved, before that fold even starts training. Once every fold is done, two comparison tables are logged: fold hyperparameters side by side, and per-fold metrics + the mean/std aggregate — the same numbers the console and local result JSON already report. Currently populated for any model that produces a training curve — `xgboost` and `residual_dnn` in the primary model set (also `dnn`/`mdn`). Needs W&B auth already set up on the machine (`wandb login`, or a `WANDB_API_KEY` env var) — the config never carries a credential.

```yaml
wandb:
  enabled: true
  project: ml-genetics4psychiatry   # default
  entity: your-team-or-username      # optional, defaults to your W&B account default
  mode: online                        # or "offline" to sync later with `wandb sync`
  tags: [xgboost, adhd]
```

→ **[documentation/4_wandb_logging.md](documentation/4_wandb_logging.md)** for what gets logged and how folds with different curve lengths (e.g. early-stopped XGBoost) are charted correctly on one plot.

### Local performance report

Every run writes a timestamped JSON to `results/{experiment_name}/{experiment_name}_YYYYMMDD_HHMMSS.json` with per-fold metrics, aggregated metrics, the full config snapshot, and (when enabled) predictions/SHAP data. Plot and aggregate saved results with [`analysis/resultAnalysis.py`](analysis/resultAnalysis.py):

```bash
python -m analysis.resultAnalysis --results results/linear_regression_scz/
python -m analysis.resultAnalysis --combined results/
```

See [`figures/README.md`](figures/README.md) for the broader figure-generation scripts built on top of saved results.

---

## 5. Feature attribution

### Why

Beyond a model's aggregate accuracy, feature attribution shows *which* input features (genotype/MRI phenotypes) drive its predictions — needed to relate a model's performance back to biologically interpretable signal, and to compare that signal across models and folds.

### Shapley values via `shapiq`

Implemented in [`src/shap.py`](src/shap.py), gated by `cfg["shap"]["enabled"]`. Every model goes through `shapiq.Explainer`, which dispatches on the estimator: a `TabularExplainer` (marginal imputation) for the sklearn linear family, a `TabPFNExplainer` for TabPFN (using its native fast re-conditioning rather than marginal imputation). Two modes, set in `shap.interactions`:

- `interactions: false` → index `"SV"`, plain Shapley values (`max_order 1`).
- `interactions: true` → index `"k-SII"`, pairwise interactions (`max_order 2`).

### Aggregation across folds

Each explained test row yields one `InteractionValues` object; those are averaged first over a fold's rows, then over the outer folds — giving one mean (signed) and one mean-magnitude attribution per feature (and per feature pair, in interaction mode) for the whole experiment. Both are written under `<results_dir>/shap/` and included in the result JSON under `shap_mean_values`. Per-row values can additionally be persisted with `shap.store: true` (independent of `predictions.enabled`) for later custom analysis — see `src/shap.py` for the full config schema.

---

## Environment setup

```bash
python3 -m venv pasteur
source pasteur/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) and [documentation/](documentation) for the full pipeline documentation.
