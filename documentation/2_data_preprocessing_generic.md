# 2. Data preprocessing — generic (`data.path` / `data.dir`)

This is the simple path: you already have one or more prepared tabular files (CSV/TSV/TXT) with numerical feature columns and a target column, and you want to run the pipeline on them without touching the GWAS-specific curation stage. It is implemented in [`main.py`](../main.py), inside the per-experiment loop.

## Required config

```yaml
data:
  path: "data/my_dataset.csv"   # a single file ...
  # dir: "data/my_datasets/"    # ... or a directory — one experiment per matching file
  target: "y"                  # exact target column name ...
  # target_regex: "^Z_scores_"  # ... or a regex matched against columns (must match exactly 1)
```

- `data.path` and `data.dir` are mutually exclusive; `data.dir` scans for files matching `data.extensions` (default `[".csv", ".tsv", ".txt"]`) and runs one experiment per file, keyed by filename stem.
- `data.target` takes priority over `data.target_regex` if both are set.

## Flow

```
data.path / data.dir
        │
        ▼
resolve file(s)                         data.dir → one experiment per matching file
        │
        ▼
load_txt_polars / load_txt              data.polars (default true), sep inferred from extension
        │
        ▼
resolve target column                   data.target, else data.target_regex (exactly 1 match)
        │
        ▼
drop_missing                            default true — df.dropna() → complete-case matrix
        │
        ▼
max_target_corr                         default: disabled (None) for this path
        │
        ▼
ignore_columns / ID split → X, y, row_ids
        │
        ▼
rand_frac row subsample                 optional, data.rand
        │
        ▼
invert                                  optional sign-flip of negative-label rows, default false
        │
        ▼
binary_classification? → see caveat below
        │
        ▼
nested_cv(X, y, model_name, cfg, ...)   → see 3_adding_a_new_model.md / README §3
```

## Defaults

| Key | Default | Notes |
|---|---|---|
| `drop_missing` | `true` | Drops any row with a missing value in any surviving column. |
| `polars` | `true` | Uses `polars` for the initial read (faster on large files). |
| `row_ratio` | `[1.0]` | Not used on this path (GWAS significance-based row filtering is specific to the specialized path). |
| `col_ratio` | `[1.0]` | Same — not used on this path. |
| `top_rows` / `top_cols` | `true` | Same — not used on this path (only consumed by `load_illness_data`). |
| `sampling` | `false` | GWAS SNP sampling — not applicable here. |
| `invert` | `false` | Sign-flips rows where the target is negative (and their features). |
| `whitening.enabled` | `false` | Requires `data.gwas_pheno_path`, which is ignored on this path — see below. |
| `residual` | `false` | Out-of-fold linear-regression residual target — works the same on this path if enabled. |

## GWAS-only keys — ignored on this path

If `data.path`/`data.dir` is set, the following keys are stripped from the effective config before anything reads them (a startup log line lists which ones were actually present and dropped):

```
illness, min_density, density_json, gwas_pheno_path, info_csv_path,
p_clump, distribution, min_complete_frac, max_target_corr
```

These belong to the specialized GWAS phenotype-panel path (see [2_data_preprocessing_specialized.md](2_data_preprocessing_specialized.md)) and are meaningless once you're pointing directly at a prepared file. In particular, `data.whitening: enabled: true` will raise a clear error on this path (it needs `gwas_pheno_path` to look up Σ by phenotype name, which is exactly the key that gets ignored).

## ⚠ Caveat: binary-classification conversion assumes a Z-score target

When `model.type` (or one entry of `model.types`) is `binary_classification`, `main.py` currently converts `y` unconditionally via:

```python
y = norm.sf(abs(y)) * 2          # two-tailed p-value from a Z statistic
y = (y <= p_value_binary).astype(int)
```

This assumes `y` is a Z-score (as GWAS summary statistics are) and turns it into a binary label by thresholding its implied p-value. **For an arbitrary generic dataset this transform is very likely not what you want** — if your target isn't a Z-score, set `model.type: regression` only, or pre-binarize your target column yourself before pointing `data.path` at it. This is a known gap in the generic path, not a documented feature — flagging it here rather than pretending it's dataset-agnostic.

## Multiple targets / multiple datasets

`data.dir` with several files, each with their own `data.target`/`data.target_regex` match, is the simplest way to run one job across several datasets — combine it with `model.names` (see README §3) to also sweep multiple models per dataset in the same run.
