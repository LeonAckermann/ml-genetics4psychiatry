# 1. Data curation

**Skip this section if the phenotype-genotype matrix already exists** (e.g. `data/pipeline/final/aligned_clumped_{ILLNESS}.txt` or a joined `gwas_pheno` matrix) — go straight to [2. Data preprocessing](2_data_preprocessing_specialized.md).

This stage turns raw per-phenotype GWAS summary-statistic files into one SNP × phenotype matrix, ready to be loaded, cleaned and split into `X`/`y` in the preprocessing stage. It is one-time work per illness/phenotype panel, orchestrated by [`main.py`](../main.py) and implemented in [`dataloader/pipeline.py`](../dataloader/pipeline.py).

**In this section:**

- This document — the two paths compared, and a full runnable curation-only config for each.
- [1_data_curation_pathA.md](1_data_curation_pathA.md) — Path A in full: stage-by-stage algorithm, config keys, batch scripts, directory layout.
- [1_data_curation_pathB.md](1_data_curation_pathB.md) — Path B in full: stage-by-stage algorithm, config keys, batch scripts, directory layout.

There are two curation paths in the codebase — use whichever matches your input files.

```
                    ┌─ Path A — single illness ─────────────────────────────────┐
                    │                                                            │
                    │  allRES.txt (per MRI phenotype)     z_{ILLNESS}.txt        │
                    │           │                              │                │
                    │           ▼                              │                │
                    │  construct_gwas_mri()                    │                │
                    │  (2-pass SNP × MRI T-stat join)           │                │
                    │           │                               ▼               │
                    │           └──────▶ aligne_illness_mri()  ──▶ plink2 clump  │
raw per-illness ────┤                          │                      │         │
GWAS files          │                          ▼                      ▼         │
                    │                  aligne_clumped_illness_mri()  (re-join)   │
                    │                          │                                 │
                    │                          ▼                                 │
                    │      aligned_clumped_{ILLNESS}.txt   (ML-ready, 1 illness) │
                    └────────────────────────────────────────────────────────────┘

                    ┌─ Path B — multi-phenotype panel ──────────────────────────┐
                    │                                                            │
                    │  z_{ILLNESS}.txt  (one file per phenotype, or per-chrom)   │
                    │           │                                                │
                    │           ▼                                                │
                    │  construct_gwas_phenotype()                                │
                    │  (join on chrom/pos or rsID; per-row allele consensus)      │
                    │           │                                                │
                    │           ▼                                                │
                    │  gwas_pheno/all_z_scores.txt  (ID + one Z column/phenotype) │
                    │           │                                                │
                    │           ▼                                                │
                    │  phenotype_clumping  (per phenotype: clump input → plink2)  │
                    │           │                                                │
                    │           ▼                                                │
                    │  output/clumped_{phenotype}.clumps  (+ optional final/)     │
                    └────────────────────────────────────────────────────────────┘
```

| | Path A | Path B |
|---|---|---|
| Scope | One illness at a time | Many phenotypes in one joined matrix |
| Config keys | `construct_gwas_mri`, `plink2` | `gwas_phenotype_construction`, `phenotype_clumping` |
| Used by | Single-illness experiments (`experiments/pipeline/pipeline_{illness}.yaml`) | `experiments/phenotypes/*.yaml` |
| Detailed doc | [1_data_curation_pathA.md](1_data_curation_pathA.md) | [1_data_curation_pathB.md](1_data_curation_pathB.md) |

## Full config — Path A

Combines both Path A stages (`construct_gwas_mri` + `plink2.prepare`) in one curation-only run (`experiment.run: false`), adapted from [`experiments/pipeline/pipeline_scz.yaml`](../experiments/pipeline/pipeline_scz.yaml):

```yaml
construct_gwas_mri:
  run: true
  input_path: "<path to the directory tree containing allRES.txt files, one per MRI phenotype>"
  output_path: "data/pipeline/input/gwas_mri/all_z_scores.txt"
  chunk_size: 10000
  total_chunks: null
  polars: false
  value: "T_STAT"

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

data:
  illness: SCZ            # a single string on this path — see the warning in 1_data_curation_pathA.md
  target: Z_scores_SCZ

experiment:
  run: false               # curation only — run training via a separate config afterward

sampling:
  run: false
```

→ Full stage-by-stage breakdown, the `data.illness` string-vs-list footgun, batch scripts, and directory layout: **[1_data_curation_pathA.md](1_data_curation_pathA.md)**.

## Full config — Path B

Combines both Path B stages (`gwas_phenotype_construction` + `phenotype_clumping`) in one curation-only run, built from the exact keys `main.py` reads for each (see [1_data_curation_pathB.md](1_data_curation_pathB.md) for a per-key breakdown):

```yaml
construct_gwas_mri:
  run: false        # Path B doesn't use the MRI-matrix stage

gwas_phenotype_construction:
  run: true
  input_path: "./data/pipeline/input/gwas_illness"
  output_path: "./data/pipeline/input/gwas_pheno/all_z_scores.txt"
  how: inner          # inner: SNP must be present in every illness file; outer: keep all SNPs
  join_key: chrom      # chrom: join on (chrom, pos) [default]; rs: join on rsID
  info_csv_path: null    # set to use the per-chromosome + include-flag input layout instead

phenotype_clumping:
  run: true
  gwas_pheno_path: "./data/pipeline/input/gwas_pheno/all_z_scores.txt"
  ref: "./data/pipeline/input/ref_panel/All_ensemble_1000G_hg38_EUR_all_chr"
  info_csv_path: "./data/pipeline/input/gwas_pheno/info.csv"
  # phenotypes: [SCZ, ADHD, BCT_BASO]     # explicit list — skips auto-selection below
  # min_density: 0.8                       # or: density-frontier auto-selection
  # density_json: "./data/pipeline/analysis/dense_density.json"
  p_clump: 1
  r2: 0.05
  clump_kb: 500
  create_final: false   # true also materializes data/pipeline/final/aligned_clumped_<phenotype>.txt

plink2:
  prepare: false        # Path B clumps per-phenotype via phenotype_clumping, not plink2.prepare

experiment:
  run: false             # curation only
```

→ Full stage-by-stage breakdown, allele-consensus/sign-flip logic, phenotype auto-selection order, batch scripts, and directory layout: **[1_data_curation_pathB.md](1_data_curation_pathB.md)**.

## Whitening pre-flight (optional, standalone)

If you plan to use `data.whitening` in preprocessing (decorrelating by a phenotype covariance/intercept matrix Σ), run the pre-flight diagnostic first — see [`whitening/README.md`](../whitening/README.md). It is not part of the `main.py` config loop.
