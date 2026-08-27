# 1b. Data curation — Path B: multi-phenotype panel pipeline

Builds one joined SNP × phenotype Z-score matrix covering *many* phenotypes at once (rather than one illness at a time), then LD-clumps each phenotype's own signal within it. This is what the current `experiments/phenotypes/*.yaml` experiments are built on. Code: [`dataloader/pipeline.py`](../dataloader/pipeline.py). Orchestrated by [`main.py`](../main.py:192) (the "One-time GWAS phenotype construction" and "Per-phenotype clumping" blocks).

```
per-illness GWAS files (single-file-per-illness, or per-chromosome + info.csv)
        │
        ▼
construct_gwas_phenotype()
  join on (chrom, pos) or rsID; per-row consensus ID/A0/A1; sign-flip on
  swapped alleles; null out unreconcilable Z values
        │
        ▼
joined gwas_pheno matrix (ID, chrom, pos, A0, A1, <phenotype_1>, <phenotype_2>, …) + info.csv
        │
        ▼
phenotype_clumping   (for each phenotype: explicit list | density-selected | info.csv include=1)
  prepare_phenotype_clump_input() → plink2 --clump → [create_final] aligne_clumped_phenotype()
        │
        ▼
data/pipeline/output/clumped_{phenotype}.clumps  (+ optional final/aligned_clumped_{phenotype}.txt)
```

## Stage 1 — join per-illness GWAS files into one phenotype matrix

**Config key:** `gwas_phenotype_construction.run: true`
**Code:** [`construct_gwas_phenotype()`](../dataloader/pipeline.py:185)

Two input layouts:

- **Single file per illness** (default, `info_csv_path` unset): every `z_<ILLNESS>.txt` under `input_path`.
- **Per-chromosome files** (`info_csv_path` set): for each `(Consortium, Outcome)` pair marked `include == 1` in `info_csv_path`, concatenates its `z_<Consortium>_<Outcome>_chr<N>.txt` files (chromosome parsed from the filename) into one phenotype named `<Consortium>_<Outcome>`. Phenotypes with no matching files are skipped.

Rows are matched across phenotypes on `join_key`:

- `"chrom"` (default) — join on `(chrom, pos)`.
- `"rs"` — join on `rsID`.

> Comparing row counts between the two modes is a useful diagnostic: a large gap (millions via `"rs"` vs. a handful via `"chrom"`) signals a genome-build mismatch between input files, since rsIDs are build-independent while chrom/pos aren't.

**Per-row allele consensus:** for each SNP, the first illness (in list order) that has data at that row supplies the reference `ID`/`A0`/`A1` — guaranteed non-null on every row, so `how="outer"` can keep rows an earlier illness lacks without ever comparing against a null reference. Every other illness's alleles are then checked against that reference:

- **Match** → kept as-is.
- **Swapped** (`A0`↔`A1` reversed) → that illness's Z-score sign is flipped.
- **Unreconcilable** → only that illness's Z-score is nulled (the row, and every other illness's value in it, survives).

Output is one wide matrix: `ID, chrom, pos, A0, A1`, then one Z-score column per illness — written with `null_value="Null"` (not empty string), which is why downstream readers need `"Null"` in their null-value list (see `GWAS_PHENO_NULL_VALS` in [`dataloader/pipeline.py`](../dataloader/pipeline.py:21) and [`documentation/2_data_preprocessing_specialized.md`](2_data_preprocessing_specialized.md)).

```yaml
gwas_phenotype_construction:
  run: true
  input_path: "./data/pipeline/input/gwas_illness"
  output_path: "./data/pipeline/input/gwas_pheno/all_z_scores.txt"
  how: inner        # inner: SNP must be present in every illness file; outer: keep all SNPs
  join_key: chrom    # chrom: join on (chrom, pos) [default]; rs: join on rsID
  info_csv_path: null  # set to switch to the per-chromosome + include-flag input layout
```

## Stage 2 — per-phenotype clumping

**Config key:** `phenotype_clumping.run: true`
**Code:** [`prepare_phenotype_clump_input()`](../dataloader/pipeline.py:453), [`call_plink2()`](../dataloader/pipeline.py:675), [`aligne_clumped_phenotype()`](../dataloader/pipeline.py:484)

The phenotype list to clump is resolved once, in priority order:

1. `phenotype_clumping.phenotypes` — an explicit list, if given.
2. Otherwise, `phenotype_clumping.min_density` + `.density_json` — the densest phenotype subset from a density-frontier JSON (see [`select_dense_features()`](../dataloader/pipeline.py:419), written by `script/alignment/dense_submatrix.py`).
3. Otherwise, every `include == 1` phenotype in `info_csv_path` that's actually present as a column in the joined matrix ([`included_phenotype_columns()`](../dataloader/pipeline.py:404)).

For each phenotype in that list:

### 2a. Build the clump input — `prepare_phenotype_clump_input()`

Reads `ID` and that phenotype's own Z-score column straight from the joined matrix (already allele-harmonized by stage 1), drops rows where it's null, and derives a two-sided p-value via `2 * norm.sf(|Z|)` (the joined matrix carries Z-scores only, no `P` column). Writes `data/pipeline/intermediate/aligned_{phenotype}{output_suffix}.txt` (`ID`, `P`).

### 2b. LD clumping — `call_plink2()`

Same `plink2 --clump` invocation as Path A stage 2b — see [documentation/1_data_curation_pathA.md](1_data_curation_pathA.md#2b-ld-clumping--call_plink2). Output: `data/pipeline/output/clumped_{phenotype}{output_suffix}.clumps`.

### 2c. Optional final materialization — `aligne_clumped_phenotype()`

Only runs when `phenotype_clumping.create_final: true`. Builds a ready-to-train "one phenotype as target" file: that phenotype's own Z column (renamed `Z`) as target, every *other* phenotype's Z column as features, rows restricted to that phenotype's own clumped SNP set. Written to `data/pipeline/final/aligned_clumped_{phenotype}{output_suffix}.txt`.

> This step is optional and usually skipped (`create_final: false`, the default) — the same "one phenotype as target, the rest as features" dataset can be built on the fly at experiment time instead, via `dataloader.preprocess.load_phenotype_clumped_data`, without ever materializing a per-phenotype file. Preferable once many phenotypes are involved, since it avoids one file per phenotype.

```yaml
phenotype_clumping:
  run: true
  gwas_pheno_path: "./data/pipeline/input/gwas_pheno/all_z_scores.txt"
  ref: "./data/pipeline/input/ref_panel/All_ensemble_1000G_hg38_EUR_all_chr"
  info_csv_path: "./data/pipeline/input/gwas_pheno/info.csv"   # used if phenotypes/min_density unset
  # phenotypes: [SCZ, ADHD, BCT_BASO]   # explicit list — skips auto-selection
  # min_density: 0.8                    # or: density-selected instead
  # density_json: "./data/pipeline/analysis/dense_density.json"
  p_clump: 1
  r2: 0.05
  clump_kb: 500
  create_final: false   # true also writes data/pipeline/final/aligned_clumped_<phenotype>.txt
```

## Batch scripts

| Script | Purpose |
|---|---|
| [`construct_gwas_phenotype.sh`](../batch_scripts/construct_gwas_phenotype.sh) | Run stage 1 |
| [`construct_gwas_phenotype_imputed.sh`](../batch_scripts/construct_gwas_phenotype_imputed.sh) | Stage 1, imputed-input variant |
| [`phenotypes/clumping.sh`](../batch_scripts/phenotypes/clumping.sh) | Run stage 2 (`phenotype_clumping.run: true`) |
| [`phenotypes/generate_shards.sh`](../batch_scripts/phenotypes/generate_shards.sh) | Split a many-phenotype training run into shards (post-curation) |

## Directory layout

```
data/pipeline/
├── input/
│   ├── gwas_illness/{z_<ILLNESS>.txt | z_<Consortium>_<Outcome>_chr<N>.txt}
│   ├── gwas_pheno/
│   │   ├── all_z_scores.txt          ← stage 1 output (joined matrix)
│   │   └── info.csv                  ← phenotype metadata (category, include flag)
│   └── ref_panel/All_ensemble_1000G_*
├── intermediate/aligned_{phenotype}.txt      ← stage 2a output
├── output/clumped_{phenotype}.clumps         ← stage 2b output
├── analysis/dense_density.json               ← density-frontier JSON (script/alignment/dense_submatrix.py)
└── final/aligned_clumped_{phenotype}.txt     ← stage 2c output, only if create_final: true
```

Downstream (Path B) training normally reads `gwas_pheno/all_z_scores.txt` + `output/clumped_{phenotype}.clumps` directly (via `data.gwas_pheno_path`), rather than the optional `final/` file — see [documentation/2_data_preprocessing_specialized.md](2_data_preprocessing_specialized.md).
