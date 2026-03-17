# ML Genetics for Psychiatry

Minimal project scaffold with data loading, model, and CLI entry point.

## Brief project description

This project studies whether MRI--genetics association signals can predict psychiatric illness risk (currently focused on SCZ) using tabular machine learning. It includes a reproducible training pipeline, multiple baseline and non-linear models, and lightweight result analysis utilities.

## Current results (SCZ, brief)

Latest combined run summary (`results/results.json`, 10 seeds, newest run per method):

- TabPFNv2 finetuned: mean $R^2 = 0.485$ (best)
- TabPFNv2: mean $R^2 = 0.440$
- Residual DNN: mean $R^2 = 0.409$
- ElasticNet / Ridge / Lasso: mean $R^2 \approx 0.392 / 0.385 / 0.382$
- XGBoost: mean $R^2 = 0.348$
- Linear Regression: mean $R^2 = 0.257$

Combined visualization is saved to `results/combined_r2_boxplot.png`.

## Project structure

- [data](data) — raw datasets (not committed)
- [dataloader](dataloader) — shared loading utilities
- [analysis](analysis) — analysis helpers (plots, result aggregation)
- [model](model) — model definitions
- [scripts](scripts) — exploratory notebooks
- [main.py](main.py) — CLI entry point

## Environment setup

```bash
python3 -m venv pasteur
source pasteur/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run an experiment

Experiments are configured via YAML files in [experiments](experiments) and run through [main.py](main.py).

```bash
python main.py --config experiments/linear_regression_scz.yaml
```

Notes:

- Run from the repo root (so relative paths like `data/...` resolve correctly).
- The first run will (by default) also generate and save deterministic train/test splits under `data/splits/{ILLNESS}_{N_SPLITS}/seed_{SEED}/`.

Available example configs:

```bash
python main.py --config experiments/ridge_regression_scz.yaml
python main.py --config experiments/lasso_regression_scz.yaml
python main.py --config experiments/elastic_regression_scz.yaml
python main.py --config experiments/xgboost_scz.yaml
python main.py --config experiments/residual_dnn_scz.yaml
```

Each run writes a timestamped JSON to `results/{experiment_name}/{experiment_name}_YYYYMMDD_HHMMSS.json` containing per-seed metrics and aggregated metrics.

### Data location (default)

The default loader expects illness-specific GWAS tables at:

- `data/tmpDATA-Leon/donnees_MRI_{ILLNESS}_only_variants_clumping_p_thr_{PVAL}all.txt`

and a target column named `Z_scores_{ILLNESS}` (e.g. `Z_scores_SCZ`). See [dataloader/dataloader.py](dataloader/dataloader.py).

## Data loading

TXT files with a header and row-wise data can be loaded with `load_txt`:

```python
from dataloader import load_txt

df = load_txt("data/your_file.txt")
```

CSV files can be loaded via `load_csv` using [dataloader/dataloader.py](dataloader/dataloader.py).

## Analysis helpers

### Quick dataset sanity-check (optional)

```python
from dataloader import load_illness_data
from analysis import basic_summary, missingness_report

# If you run this from the repo root:
df = load_illness_data("SCZ", in_notebook=False)

print(basic_summary(df))
print(missingness_report(df).head(20))
```

### Evaluate results (metrics + plots)

To plot R² distributions from saved result JSONs:

```bash
# One method (all runs in that folder)
python -m analysis.resultAnalysis --results results/linear_regression_scz/

# One boxplot per method (newest run per method)
python -m analysis.resultAnalysis --combined results/
```

By default the plots are saved next to the results:

- `results/<method>/r2_boxplot.png`
- `results/combined_r2_boxplot.png`

## Notebook quickstart

Start with one of:

- `scripts/notebooks_old/quickstart.ipynb`
- `scripts/data/data.ipynb`
- `scripts/data/graph.ipynb`

## Requirements

See [requirements.txt](requirements.txt).
