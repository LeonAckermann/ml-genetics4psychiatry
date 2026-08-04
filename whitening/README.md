# `whitening/` — pre-flight diagnostics before whitening a phenotype covariance matrix

## The reframe this module is built around

"kappa needs to be low" is not the right question. If Sigma were known
exactly, kappa would be irrelevant: the lambda_i in the numerator of
`(v_i' z)^2` cancels the `lambda_i^{-1}` in the whitener exactly, and a
correctly-specified whitening of a matrix with kappa = 1e6 is perfectly
calibrated. What matters is the ratio of estimation error to the smallest
eigenvalue, `||E||_2 / lambda_min`. kappa is a *proxy* for that ratio, not the
quantity itself.

So the pre-flight question isn't "is kappa small" but "is the bottom of my
spectrum estimated well enough to divide by" — and every check in this module
exists to answer some piece of that.

## Quick start

```bash
# Fast, self-contained toy target (2 phenotypes, kappa ~= 3.4, generated on
# the fly — no external data needed):
python -m whitening.toy --out-dir data/pipeline/input/whitening_toy
python -m whitening.run --config whitening/configs/toy.yaml

# The real 192-trait LDSC/JASS intercept matrix — same checklist, one config
# swap:
python -m whitening.run --config whitening/configs/full_intercept_matrix.yaml
```

Each run writes `results/whitening/<name>/whitening_report.{json,txt}` plus
`spectrum.png`, `calibration_tail.png`, `sweep.png`. Exit code is `1` if any
check fails, `2` if the config/inputs couldn't even be loaded, `0` otherwise —
chain it with `&&` in a batch script to gate a downstream stage on a clean
report.

Scaling to a new matrix is a config change, not a code change: copy
`configs/full_intercept_matrix.yaml`, point `sigma.path` at the new file, and
fill in `z.path` / `probe.sigma2_path` / `sign_check.groups_path` as available
— each is optional and the corresponding check reports `SKIP` (not a
silent no-op) when its input is missing.

## What each check does, and why

| # | Check | Module | Needs | What it catches |
|---|-------|--------|-------|------------------|
| 0 | Two-estimate error probe | `probe.py` | a second Sigma estimate | `\|\|Sigma1-Sigma2\|\|_2` vs `lambda_min` directly — the single most informative number here (see caveat below) |
| 1 | Column alignment | `structure.py` | a Z matrix | Sigma's trait order not matching Z's columns — a permutation mismatch is still square, still PD, still whitens *something*, just not your data. **Only ever reorders Sigma (lossless); a partial Z never shrinks it** — every other check still runs at full scope |
| 2 | Symmetry | `structure.py` | — | `eigh` reads one triangle silently; pairwise-estimated entries rarely agree to machine precision |
| 3 | Missing / defaulted entries | `structure.py` | — | NaNs, and exact-zero off-diagonals (the JASS <1000-qualifying-SNP default) — a modelling choice masquerading as a measured value |
| 4 | Diagonal | `structure.py` | — | correlation form (diag=1, ZCA==ZCA-cor) vs raw intercepts `1+a_k` (V != I, transform choice is no longer moot) |
| 5 | Full spectrum | `spectrum.py` | — | clean bulk + outliers (truncation has a natural cutoff) vs smooth decay (it doesn't — ridge is more honest); reports effective rank and trace share in the bottom decile |
| 6 | Positive definiteness | `spectrum.py` | — | `lambda_min <= 0` is not a rounding artifact to clip past — it's direct evidence that `\|\|E\|\|_2 > lambda_min^true` |
| 7 | Sign consistency | `sign.py` | a trait-group map | scattered sign flips among strongly-correlated same-group pairs — the fingerprint of unresolved strand-ambiguous (A/T, C/G) alleles, which attenuate off-diagonals and inflate `lambda_min` artificially |
| 7b | Strand-ambiguous SNP fraction | `sign.py` | Z + allele columns | direct count of A/T, C/G SNPs in the loaded data |
| 8 | Null calibration | `calibration.py` | — | the acid test: whiten simulated `z ~ N(0, Sigma)`, check `Cov(w) ~= I`, `E[\|w\|^2] ~= K`, and the `chi^2_K` **tail** (mean calibration can look fine while the 1e-8 quantile is off by orders of magnitude) |
| — | Regularisation sweep | `calibration.py` | — | sweeps ridge/floor/Higham strength, recommends the *smallest* alpha that achieves full null calibration — turns an arbitrary numerical fix into a reported, tuned parameter |

Checks that need an optional input (Z, a second Sigma, a group map) report
`SKIP` with an explicit message when it's absent, rather than being silently
omitted from the report.

### The two-estimate probe's caveat (read before trusting check 0)

`||Sigma1 - Sigma2||_2` is only a clean noise estimate when both matrices
target the *same estimand* and differ by sampling error — e.g. two
null-SNP-filter variants, or LDSC on two SNP subsets. If instead one is a
naive empirical z-correlation and the other is LDSC/JASS-filtered, the
difference is dominated by **bias** (the `l_bar * Sigma_g` contamination), not
noise, and the probe overstates the estimation error. Set
`probe.same_estimand: false` in that case — the report's message changes to
flag it as an upper bound rather than a noise estimate. See the docstring in
`probe.py` for the full reasoning.

## Config reference

See `configs/toy.yaml` for an annotated, working example and
`configs/full_intercept_matrix.yaml` for the same checklist pointed at
`data/pipeline/input/gwas_pheno/official_intercept_matrix.csv` (192 LDSC/JASS
traits), with inline notes on what's different at that scale. Top-level keys:

* `sigma.path` (required) — the matrix to whiten by. `.csv`/`.tsv`/`.parquet`
  with row+column labels, or `.npy` + `sigma.labels_path`.
* `z` (optional) — SNP x trait z-score matrix, for checks 1, 7b, and to
  simulate/validate against real column structure. `z.max_snps` thins large
  files with a deterministic stride so a 9M-row matrix never loads whole.
* `probe.sigma2_path` (optional) — see check 0 above.
* `sign_check.groups_path` (optional) — CSV mapping trait -> group (e.g.
  `reference.csv`'s `Category`). `key_col` can be a list of columns to compose
  a composite key (needed because `official_intercept_matrix.csv`'s labels are
  `{Consortium}_{Outcome}` but `reference.csv` only has them split).
* `transform.standardize` — `auto` (rescale to correlation form only if the
  diagonal check warned) / `true` / `false`.
* `transform.transform` — `zca` / `zca-cor` / `pca` / `cholesky` (see
  `spectrum.py::build_whitener`'s docstring for what differs between them).
* `regularization` — the method (`ridge` / `floor` / `higham` / `none`) and
  strength used for the calibration check (0 auto-floors just enough to reach
  PD if needed; the `sweep` block is what should actually decide the value).
* `sweep.run: true` — enables the regularisation-strength sweep.
* `report.out_dir` / `report.plots` — where results land.

## What actually gets flagged on the real matrix

Running `configs/full_intercept_matrix.yaml` against the current
`official_intercept_matrix.csv` (192 traits) reports **FAIL** overall:

* diagonal ranges 0.85 - 3.23 (raw LDSC intercepts, not correlations — 24
  traits sit *below* 1, which `1 + a_k` with `a_k >= 0` can't do, so that's
  itself estimation noise)
* 188 off-diagonal entries are exact zero (0.51%) — the JASS
  <1000-qualifying-SNP default written into the matrix
* **not positive semi-definite**: 8 negative eigenvalues, `lambda_min = -0.235`
* 6% of strong within-Category pairs have a sign opposite their group's
  majority (check 7)
* the regularisation sweep needs `alpha = 0.5` (ridge, correlation form)
  before null calibration passes

None of this is a bug in the loader — it's the actual state of that matrix,
which is exactly what this module exists to surface before it gets inverted.
