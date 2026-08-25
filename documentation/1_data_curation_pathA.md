# 1a. Data curation — Path A: single-illness pipeline

Builds one illness's ML-ready feature matrix from a per-MRI-phenotype GWAS result directory. Code: [`dataloader/pipeline.py`](../dataloader/pipeline.py). Orchestrated by [`main.py`](../main.py:178) (the "One-time GWAS MRI construction" and "Data pipeline" blocks).

```
allRES.txt (per MRI phenotype)          z_{ILLNESS}.txt (illness GWAS)
        │                                        │
        ▼                                        │
┌────────────────────────┐                       │
│ construct_gwas_mri()   │                       │
│  SNP × MRI T-stat       │                       │
│  matrix (2-pass join)   │                       │
└───────────┬─────────────┘                       │
            ▼                                     ▼
        all_z_scores.txt  ──────▶ aligne_illness_mri()  (direct + flipped-allele match)
                                          │
                                          ▼
                              aligned_{ILLNESS}.txt ──▶ plink2 --clump (LD clumping)
                                          │
                                          ▼
                              clumped_{ILLNESS}.clumps
                                          │
                                          ▼
                          aligne_clumped_illness_mri()  (re-join + drop plink2 metadata)
                                          │
                                          ▼
                    data/pipeline/final/aligned_clumped_{ILLNESS}.txt   (ML-ready)
```

## Stage 1 — merge per-phenotype GWAS files into one SNP × MRI matrix

**Config key:** `construct_gwas_mri.run: true`
**Code:** [`construct_gwas_mri()`](../dataloader/pipeline.py:23)
**Batch scripts:** [`construct_gwas_mri.sh`](../batch_scripts/construct_gwas_mri.sh), [`construct_gwas_mri_p.sh`](../batch_scripts/construct_gwas_mri_p.sh)

Two-pass join over every `allRES.txt` file under `input_path`:

1. **Pass 1 — find common SNPs.** Each file is scanned for `(ID, A1, OMITTED)`; the intersection across every file is the common-SNP reference frame.
2. **Pass 2 — extract T-statistics.** For each file: detect duplicate `(ID, A1, OMITTED)` rows (logged to `duplicate_snps.tsv`), keep the one with the largest `|T_STAT|`, then join onto the common-SNP frame and write its `T_STAT` column into a pre-allocated `(n_common_snps × n_files)` matrix.

Allele columns are then remapped (`A1 → A0`, `OMITTED → A1`) to match the illness-GWAS convention, and the matrix is written tab-separated.

```yaml
construct_gwas_mri:
  run: true
  input_path: "<path to the directory tree containing allRES.txt files, one per MRI phenotype>"
  output_path: "data/pipeline/input/gwas_mri/all_z_scores.txt"
  chunk_size: 10000
  total_chunks: null      # cap how many allRES.txt files are read, for a quick test run
  polars: false
  value: "T_STAT"          # which column of allRES.txt becomes the matrix's cell values
```

## Stage 2 — align illness GWAS with the MRI matrix and LD-clump

**Config key:** `plink2.prepare: true`
**Code:** [`aligne_illness_mri()`](../dataloader/pipeline.py:554), [`call_plink2()`](../dataloader/pipeline.py:675), [`aligne_clumped_illness_mri()`](../dataloader/pipeline.py:694)

### 2a. First alignment — `aligne_illness_mri()`

Reads `data/pipeline/input/gwas_illness/z_{illness}.txt` and the MRI matrix from stage 1 (`plink2.mri`), then aligns them via [`merge_gwas_illness_mri()`](../dataloader/pipeline.py:528):

1. **Direct match:** inner-join on `[ID, A0, A1]`.
2. **Flipped-allele match:** for rows the direct join missed, swap `A0 ↔ A1` and negate `Z`. Palindromic SNPs (`A`/`T` or `C`/`G` allele pairs) are excluded first, since strand orientation can't be resolved for them.
3. **Concatenate** direct + flipped matches.

Only `ID` and `P` are kept and written to `data/pipeline/intermediate/aligned_{illness}{output_suffix}.txt` — the plink2 clumping input.

### 2b. LD clumping — `call_plink2()`

Shells out to the `plink2` binary (resolved via the `PLINK2` env var, `PATH`, or `~/tools/{bin,plink2}/plink2` — see `_resolve_plink2()`):

```
plink2 \
  --bfile <plink2.ref>       \
  --clump <plink2.aligned>   \
  --clump-kb <plink2.clump_kb>   \
  --clump-r2 <plink2.r2>         \
  --clump-p1 <plink2.p_clump>    \
  --clump-p2 <plink2.p_clump>    \
  --out <plink2.output>
```

| Parameter | Purpose | Typical value |
|---|---|---|
| `--bfile` | Reference panel (plink bfile triple) — e.g. 1000 Genomes EUR (hg38) | required |
| `--clump-r2` | LD r² threshold; SNP pairs above this are pruned | `0.05` |
| `--clump-kb` | LD window in kilobases | `500` |
| `--clump-p1` / `--clump-p2` | p-value thresholds for index/secondary SNPs | `1` (keep all) |

Output: `<plink2.output>.clumps` — the independent SNP set.

### 2c. Second alignment — `aligne_clumped_illness_mri()`

1. Loads the `.clumps` file.
2. Re-aligns illness GWAS with the MRI matrix (same logic as 2a).
3. Inner-joins the clumped SNP list with that aligned data on `ID`.
4. Drops plink2 metadata columns: `#CHROM`, `POS`, `TOTAL`, `NONSIG`, `S0.05`, `S0.01`, `S0.001`, `S0.0001`, `SP2`, plus the illness-side `chrom`, `pos`, `A0`, `A1`, `N`.

Output: `data/pipeline/final/aligned_clumped_{illness}{output_suffix}.txt` — the ML-ready feature matrix.

## Config — full example (curation-only run)

Real config, drawn from [`experiments/pipeline/pipeline_scz.yaml`](../experiments/pipeline/pipeline_scz.yaml). Note `experiment.run: false` — this config only runs curation; a separate config (with `data.illness` as a *list*, see the warning below) is used for training afterward.

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
  # output_suffix: "_phenotype"   # set this to point `mri` at a different matrix
                                   # (e.g. the joined gwas_pheno matrix from Path B)
                                   # without colliding with a default-suffix run

experiment:
  run: false      # this config only runs curation, not training

sampling:
  run: false

data:
  illness: SCZ                # a single string here — see warning below
  target: Z_scores_SCZ
```

### ⚠ `data.illness` means two different things in two different stages

In this stage (`plink2.prepare`), `data.illness` is read as **one plain string** (`aligne_illness_mri(illness=data_cfg["illness"])`). In the training-loop stage (`experiment.run: true`, see [documentation/2_data_preprocessing_specialized.md](2_data_preprocessing_specialized.md)), `data.illness` is instead expected to be **a list** (`illness_list = data_cfg.get("illness") or []`, then looped over). If you reuse the same config for both stages, or set `data.illness: SCZ` (a bare string) intending it for the training loop, `main.py` will iterate over the individual *characters* of `"SCZ"` instead of the illness name — use `data.illness: [SCZ]` for any config where `experiment.run: true`.

## Batch scripts

| Script | Purpose |
|---|---|
| [`construct_gwas_mri.sh`](../batch_scripts/construct_gwas_mri.sh) | Run stage 1 only |
| [`pipeline.sh`](../batch_scripts/pipeline.sh) | Run stage 1 + 2 in sequence via a `pipeline_*.yaml` config |

## Directory layout

```
data/pipeline/
├── input/
│   ├── gwas_mri/all_z_scores.txt         ← stage 1 output
│   ├── gwas_illness/z_{ILLNESS}.txt      ← raw illness GWAS summary stats
│   └── ref_panel/All_ensemble_1000G_*    ← reference panel (plink bfile)
├── intermediate/aligned_{ILLNESS}.txt    ← stage 2a output
├── output/clumped_{ILLNESS}.clumps       ← stage 2b output
└── final/aligned_clumped_{ILLNESS}.txt   ← stage 2c output (ML-ready)
```
