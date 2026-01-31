# ML Genetics for Psychiatry

Minimal project scaffold with data loading, model, and CLI entry point.

## Project structure

- [data](data) — raw datasets (not committed)
- [dataloader](dataloader) — shared loading utilities
- [datanalysis](datanalysis) — analysis helpers
- [model](model) — model definitions
- [scripts/notebooks](scripts/notebooks) — exploratory notebooks
- [main.py](main.py) — CLI entry point

## Environment setup

```bash
python3 -m venv pasteur
source pasteur/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run an experiment

```bash
python main.py --data data/your_data.csv --target your_target_column
```

## Data loading

TXT files with a header and row-wise data can be loaded with `load_txt`:

```python
from dataloader import load_txt

df = load_txt("data/your_file.txt")
```

CSV files can be loaded via `load_csv` using [dataloader/dataloader.py](dataloader/dataloader.py).

## Analysis helpers

Common analysis helpers live in [datanalysis/analysis.py](datanalysis/analysis.py). Add new functions there so they can be imported from notebooks and scripts.

## Notebook quickstart

Open the notebook at scripts/notebooks/quickstart.ipynb.

## Requirements

See [requirements.txt](requirements.txt).
