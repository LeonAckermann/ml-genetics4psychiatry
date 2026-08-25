# 2. Data preprocessing — specialized (GWAS phenotype-genotype matrix)

Used when `data.path`/`data.dir` are **not** set — the pipeline instead resolves an illness/phenotype-specific file from `data/sampled/{distribution}/sampled_{illness}_p{p}.txt` (or triggers on-the-fly SNP sampling to produce it). This path exposes every GWAS-specific knob. Implemented in [`main.py`](../main.py), same per-experiment loop as the generic path — the two only diverge at data loading and a handful of GWAS-only steps.

## Flow

```
data.illness (list, or auto-detected via min_density/density_json or info.csv)
  × data.distribution × data.p_clump                (product() over all three, plus model/task/noise/... axes)
        │
        ▼
data.sampling: true?
   ├─ yes → sample()  (dataloader/preprocess.py)
   │         uninformed | uniform SNP sampling strategy
   │         + feature pruning: max_target_corr, min_complete_frac / max_col_missing
   │         → writes data/sampled/{distribution}/sampled_{illness}_p{p}.txt
   └─ no  → reuse existing sampled_{illness}_p{p}.txt as-is (must already exist)
        │
        ▼
load_illness_data()                     row_ratio / col_ratio / top_rows / top_cols significance-based
                                         feature & row pruning, mri_p_value
        │
        ▼
exclude_same_category                   optional — drops predictor phenotypes sharing the target's
                                         GWAS reference-sheet trait category, before any other pruning
        │
        ▼
resolve target column                   data.target or data.target_regex
        │
        ▼
drop_missing                            default true
        │
        ▼
whitening                               optional — decorrelates target + features by Σ (LDSC/JASS
                                         intercept covariance); requires data.gwas_pheno_path
        │
        ▼
max_target_corr                         default 0.8 — drops features too correlated with the target
                                         (re-checked here even if sampling already applied it)
        │
        ▼
ignore_columns / ID split → X, y, row_ids
        │
        ▼
rand_frac row subsample → invert → binary_classification norm.sf(|y|) conversion
        │                                (correct here — y genuinely is a Z-score)
        ▼
data.residual: true?  →  out-of-fold LinearRegression residual becomes the new y
        │                (see src/cv.py::_residual_target — same outer folds as the real model)
        ▼
nested_cv(X, y, model_name, cfg, ...)
```

## Config keys specific to this path

| Key | Purpose | Default |
|---|---|---|
| `data.illness` | List of phenotypes/illnesses to loop over | auto-detected from `info.csv` (or density-selected via `min_density`) if empty |
| `data.min_density` + `data.density_json` | Auto-select phenotypes at/above a feature-density threshold instead of listing them | — |
| `data.gwas_pheno_path` | Path to the joined GWAS phenotype matrix | required for sampling, whitening, phenotype auto-detection |
| `data.info_csv_path` | Phenotype metadata (category, inclusion flag) | required for phenotype auto-detection |
| `data.p_clump` | LD-clumping p-value threshold(s) to loop over | — |
| `data.distribution` | SNP sampling distribution(s) (`uninformed`/`uniform`) to loop over | — |
| `data.sampling` | Recompute the sampled file (vs. reuse an existing one) | `false` |
| `data.min_complete_frac` / `data.max_col_missing` | Feature-density pruning during sampling | — |
| `data.max_target_corr` | Drop features whose Pearson `|r|` with the target exceeds this | `0.8` |
| `data.row_ratio` / `data.col_ratio` | Fraction of significance-ranked rows/columns kept by `load_illness_data` | `[1.0]` |
| `data.top_rows` / `data.top_cols` | Keep the *most* significant rows/columns (vs. least) | `true` |
| `data.exclude_same_category` | Drop predictor phenotypes in the same GWAS trait category as the target | `false` |
| `data.whitening` | Σ-decorrelation of target + features (see below) | disabled |
| `data.residual` | Replace `y` with its out-of-fold linear-regression residual | `false` |

All of the *generic*-path keys (`drop_missing`, `polars`, `invert`, `ignore_columns`, `rand`/`rand_frac`, `noise.sigma`, `data.pca`) apply identically here — see [2_data_preprocessing_generic.md](2_data_preprocessing_generic.md) for those.

## Whitening

Decorrelates cross-trait nuisance correlation (sample overlap, population structure) captured by Σ, the LDSC/JASS phenotype intercept-covariance matrix, before any other feature pruning runs. Requires `data.gwas_pheno_path` (Σ's trait labels are matched against phenotype-panel column names). Run the [whitening pre-flight diagnostic](../whitening/README.md) once against your Σ matrix before relying on it — see [1_data_curation.md](1_data_curation.md#whitening-pre-flight-optional-standalone).

```yaml
data:
  whitening:
    enabled: true
    sigma_path: "data/pipeline/input/gwas_pheno/official_intercept_matrix.csv"
    method: ridge            # or another supported regularization method
    transform_method: zca    # or another supported whitening transform
```

## Residual target

`data.residual: true` fits a plain out-of-fold `LinearRegression` per outer fold (same fold split the real model is scored on) and replaces `y` with its residual before the real model trains — isolates what the real model adds beyond a linear baseline on the same features. Only defined for regression (`nested_cv` raises if combined with `binary_classification`). The linear baseline's own aggregated metrics are saved alongside the real model's under `residual_baseline` in the result JSON, for direct comparison.

## Full example — `data:` section

Adapted from [`experiments/phenotypes/linear_regression_all_phenotypes.yaml`](../experiments/phenotypes/linear_regression_all_phenotypes.yaml), exercising most of the keys above:

```yaml
data:
  # illness blank -> auto-detected. With min_density set, the phenotype list is
  # taken from the density-frontier JSON (the same features used for clumping)
  # instead of info_csv_path; unset min_density to fall back to info.csv.
  illness: []
  min_density: 0.8
  density_json: "./data/pipeline/analysis/dense_density.json"
  gwas_pheno_path: "./data/pipeline/input/gwas_pheno/all_z_scores_imputed.txt"
  info_csv_path: "./data/pipeline/input/gwas_pheno/info.csv"
  p_clump: [0.001]
  distribution: ["low"]
  target: Z
  # Feature-column pruning during sampling (drops sparse phenotype columns so the
  # complete-case matrix keeps rows). Use ONE of the following:
  #   min_complete_frac: auto-pick the densest features so that >= this fraction
  #                      of rows stay complete (takes precedence if both set).
  #   max_col_missing:   drop any feature column whose null-rate exceeds this.
  min_complete_frac: 0.95
  # Marginal |Pearson r| with the target. Applied twice, independently: inside
  # sampling (before the densest-feature selection above) and again on the
  # loaded matrix before training, so it holds with sampling: false too.
  # Set to null to disable both.
  max_target_corr: 0.8
  drop_missing: true   # then drop the residual rows with any null before training
  polars: true
  chunk_size: 100000
  row_ratio: [1.0]
  col_ratio: [1.0]
  top_rows: true
  top_cols: true
  # min_complete_frac / max_col_missing only run inside the sampling stage; with
  # sampling: false the existing data/sampled/<dist>/sampled_<illness>_p<p>.txt is
  # reused as written. max_target_corr applies either way (see above).
  sampling: false
  invert: true
  whitening:
    enabled: true
    sigma_path: data/pipeline/input/gwas_pheno/covariance_spd.csv   # pre-regularized Sigma (SPD) for ZCA whitening
    search_alpha: false   # use the pre-regularized Sigma as-is, no per-target null search
    transform_method: zca
```
