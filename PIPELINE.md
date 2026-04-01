# Pipeline – Inner Workings

This document describes the data-processing and training pipeline that turns raw GWAS summary statistics into ML-ready datasets and experiment results.

## Overview

The pipeline is orchestrated by [`main.py`](main.py), which reads a YAML config and conditionally runs up to four stages:

```
1. Construct GWAS–MRI matrix       (construct_gwas_mri)
2. Align & clump SNPs via plink2   (plink2.prepare)
3. Sample SNPs by p-value          (sampling)
4. Train & evaluate ML models      (experiment)
```

Each stage can be toggled independently through the YAML config. Stages 1–3 are data-preparation steps that only need to run once per illness; stage 4 is the repeatable training loop.

## 1. Construct GWAS–MRI matrix

**Config key:** `construct_gwas_mri.run: true`
**Code:** [`dataloader/pipeline.py :: construct_gwas_mri()`](dataloader/pipeline.py)
**Batch script:** [`batch_scripts/construct_gwas_mri.sh`](batch_scripts/construct_gwas_mri.sh)

This one-time step merges per-MRI-phenotype GWAS result files (`allRES.txt`) into a single SNP × phenotype T-statistic matrix.

### Algorithm (two-pass)

**Pass 1 – Find common SNPs.**
Scan every `allRES.txt` file under the input directory. Each SNP is keyed by `(ID, A1, OMITTED)`. The intersection of keys across all files gives the set of SNPs present in every phenotype.

**Pass 2 – Extract T-statistics.**
Pre-allocate a `(n_common_snps × n_files)` NumPy matrix. For each file:
  - Detect and log duplicate SNP entries (kept in `duplicate_snps.tsv`).
  - De-duplicate by keeping the entry with the largest absolute T-statistic.
  - Join on the common-SNP reference frame and write the T-statistic column into the matrix.

Finally, allele columns are remapped (`A1 → A0`, `OMITTED → A1`) and the full matrix is written as a tab-separated file (default: `data/pipeline/input/gwas_mri/all_z_scores.txt`).

### Typical config

```yaml
construct_gwas_mri:
  run: true
  input_path: "<path to pheno_gwas directory containing allRES.txt files>"
  output_path: "data/pipeline/input/gwas_mri/all_z_scores.txt"
  polars: true
  chunk_size: 100000
```

## 2. Align illness GWAS with MRI data & plink2 clumping

**Config key:** `plink2.prepare: true`
**Code:** [`dataloader/pipeline.py`](dataloader/pipeline.py) — `aligne_illness_mri()`, `call_plink2()`, `aligne_clumped_illness_mri()`

This stage runs three sub-steps sequentially:

### 2a. First alignment — `aligne_illness_mri()`

Aligns illness GWAS z-scores (e.g. `z_SCZ.txt`) with the MRI z-score matrix on the SNP allele key `(ID, A0, A1)`.

1. **Direct match:** Inner-join illness and MRI data on `[ID, A0, A1]`.
2. **Flipped-allele match:** For unmatched SNPs, swap `A0 ↔ A1` and negate the Z-score. Palindromic SNPs (A/T or C/G pairs) are excluded because strand orientation is ambiguous.
3. **Concatenate** direct and flipped matches.

The aligned file (only `ID` and `P` columns) is saved to `data/pipeline/intermediate/aligned_{ILLNESS}.txt` for use as the plink2 clumping input.

### 2b. LD clumping — `call_plink2()`

Invokes the external `plink2` binary to perform linkage-disequilibrium (LD) clumping:

```
plink2 \
  --bfile <reference panel>  \
  --clump <aligned file>     \
  --clump-kb 500             \
  --clump-r2 0.05            \
  --clump-p1 <p_clump>       \
  --clump-p2 <p_clump>       \
  --out <output prefix>
```

| Parameter | Purpose | Default |
|---|---|---|
| `--bfile` | 1000 Genomes EUR reference panel (hg38) | Required |
| `--clump-r2` | LD r² threshold; SNP pairs above this are pruned | 0.05 |
| `--clump-kb` | LD window in kilobases | 500 |
| `--clump-p1/p2` | p-value thresholds for index/secondary SNPs | 1 (keep all) |

Output: `clumped_{ILLNESS}.clumps` containing the independent SNP set.

### 2c. Second alignment — `aligne_clumped_illness_mri()`

Merges the clumped SNP list back with the full aligned illness–MRI data:

1. Load the `.clumps` file from plink2.
2. Re-align illness GWAS with MRI (same allele-matching logic as 2a).
3. Inner-join clumped SNPs with the aligned data.
4. Drop plink2 metadata columns (`#CHROM`, `POS`, `TOTAL`, `NONSIG`, `S0.05`, `S0.01`, `S0.001`, `S0.0001`, `SP2`, and the illness-side `chrom`, `pos`, `A0`, `A1`, `N`).

Output: `data/pipeline/final/aligned_clumped_{ILLNESS}.txt` — the ML-ready feature matrix.

### Typical config

```yaml
plink2:
  prepare: true
  p_clump: 1
  r2: 0.05
  clump_kb: 500
  chunk_size: 100000
  mri: "./data/pipeline/input/gwas_mri/all_z_scores.txt"
  ref: "./data/pipeline/input/ref_panel/All_ensemble_1000G_hg38_EUR_all_chr"
  aligned: "./data/pipeline/intermediate/aligned_SCZ.txt"
  output: "./data/pipeline/output/clumped_SCZ"
  polars: true
```

## 3. SNP sampling

**Config key:** `sampling.run: true`
**Code:** [`dataloader/preprocess.py :: sample()`](dataloader/preprocess.py)

Optional stage that sub-samples SNPs from the aligned/clumped data for faster iteration or distribution-balancing experiments.

Two sampling strategies are available:

| Strategy | Description |
|---|---|
| `uninformed` | Keep all genome-wide-significant SNPs (p < threshold) and randomly sample `sample_size` SNPs from the non-significant set. |
| `uniform` | Bin non-significant SNPs by Z-score into equal-frequency bins (default 100). Sample a minimum count from each bin to produce a nearly uniform Z-score distribution. |

Output: `data/sampled/{distribution}/sampled_{ILLNESS}_p{P_VALUE}.txt`

## 4. Train & evaluate

**Config key:** `experiment.run: true`
**Code:** [`main.py`](main.py)

### Data loading

The experiment stage loads the final clumped data via `load_illness_data()` ([`dataloader/dataloader.py`](dataloader/dataloader.py)), which reads:

```
data/tmpDATA-Leon/donnees_MRI_{ILLNESS}_only_variants_clumping_p_thr_{PVAL}all.txt
```

The target column is `Z_scores_{ILLNESS}`. Illness-specific p-value thresholds:
- SCZ: 0.0001
- MDD, ADHD, ASD, OCD, BIP: 0.001

### Train/test splits

`prepare_data_splits()` generates `n_splits` reproducible splits (seeds 42, 43, …) with `StandardScaler` fitted on each training fold. Splits are saved as `.npz` files under `data/splits/{ILLNESS}_{N_SPLITS}/seed_{SEED}/`.

### Supported models

| Model | Config name | Key hyperparameters |
|---|---|---|
| OLS | `linear_regression` | — |
| Ridge | `ridge_regression` | `best_alpha` |
| Lasso | `lasso_regression` | `best_alpha` |
| ElasticNet | `elastic_regression` | `best_alpha`, `best_l1_ratio` |
| XGBoost | `xgboost` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |
| DNN | `dnn` | `hidden_dims`, `dropout`, `learning_rate`, `epochs`, `batch_size` |
| Residual DNN | `residual_dnn` | same as DNN |

**DNN training details:**
- Loss: MSELoss
- Optimizer: Adam with gradient clipping (max_norm=1.0)
- LR scheduler: CosineAnnealingWarmRestarts (T₀=15, T_mult=2, η_min=1e-6)
- Early stopping: triggered after 20 consecutive epochs where validation loss increases
- Best model checkpoint: restored from the epoch with lowest validation loss

### Evaluation

Metrics computed per seed and aggregated across all seeds:
- **R²** (coefficient of determination)
- **MSE** (mean squared error)
- Optional binary classification metrics (accuracy, precision, recall, F1) when `binary_threshold` is set.

### Output

Each run writes a timestamped JSON to `results/{experiment_name}/{experiment_name}_YYYYMMDD_HHMMSS.json` containing:
- `gwas_mri_stats` — merge statistics (if stage 1 ran)
- `illness_mri_alignment` — first alignment counts
- `plink2` — clumping parameters used
- `clumped_illness_mri_alignment` — second alignment counts
- `sampling_metrics` — sampling statistics (if stage 3 ran)
- `per_seed` — per-split metrics
- `aggregated` — metrics aggregated over all seeds
- `config` — full YAML config snapshot

## Batch scripts (SLURM)

All batch scripts are in [`batch_scripts/`](batch_scripts/) and target a SLURM cluster.

| Script | Purpose | Key resources |
|---|---|---|
| [`pipeline.sh`](batch_scripts/pipeline.sh) | Run the full pipeline (loads Python 3.12 + plink2, executes `main.py --config experiments/pipeline_scz.yaml`) | 200 GB RAM, 64 CPUs |
| [`construct_gwas_mri.sh`](batch_scripts/construct_gwas_mri.sh) | Run only the MRI matrix construction step | 200 GB RAM, 12 CPUs |
| [`data.sh`](batch_scripts/data.sh) | Run auxiliary data scripts | 10 GB RAM |
| [`tabpfn_mdd.sh`](batch_scripts/tabpfn_mdd.sh) | TabPFN inference for MDD (GPU) | 100 GB RAM, A100 GPU |
| [`finetune_tabpfn_mdd.sh`](batch_scripts/finetune_tabpfn_mdd.sh) | TabPFN fine-tuning for MDD (GPU) | GPU partition |
| [`unzip.sh`](batch_scripts/unzip.sh) | Extract raw GWAS archives | — |

## Directory layout (data flow)

```
data/pipeline/
├── input/
│   ├── gwas_mri/
│   │   └── all_z_scores.txt         ← stage 1 output (SNP × MRI T-stats)
│   ├── gwas_illness/
│   │   └── z_{ILLNESS}.txt          ← raw illness GWAS summary stats
│   └── ref_panel/
│       └── All_ensemble_1000G_*     ← 1000G EUR reference (plink bfile)
├── intermediate/
│   └── aligned_{ILLNESS}.txt        ← stage 2a output (ID + P for plink2)
├── output/
│   └── clumped_{ILLNESS}.clumps     ← stage 2b output (independent SNPs)
└── final/
    └── aligned_clumped_{ILLNESS}.txt ← stage 2c output (ML-ready matrix)
```

## Experiment YAML configs

Pre-built pipeline configs exist for multiple illnesses:

```
experiments/pipeline_scz.yaml
experiments/pipeline_mdd.yaml
experiments/pipeline_adhd.yaml
experiments/pipeline_asd.yaml
experiments/pipeline_ocd.yaml
experiments/pipeline_bip.yaml
experiments/pipeline_az.yaml
experiments/construct_gwas_mri.yaml
```

Model-specific configs (training only, no data processing):

```
experiments/linear_regression_scz.yaml
experiments/ridge_regression_scz.yaml
experiments/lasso_regression_scz.yaml
experiments/elastic_regression_scz.yaml
experiments/xgboost_scz.yaml
experiments/residual_dnn_scz.yaml
experiments/uniform_scz.yaml
```
