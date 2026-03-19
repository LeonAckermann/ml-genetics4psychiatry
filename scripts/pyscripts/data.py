from pathlib import Path
import sys
import torch
import torch

# Ensure project root is on sys.path for imports (must be before local imports)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from dataloader import load_txt, preprocess, GWASDataset

#print working directory
print("Current working directory:", Path.cwd())

data_path = Path("../data/tmpDATA-Leon/data_clumping_and_sumstats_for_leon/donnees_MRI_et_diseases.txt")

df = load_txt(data_path, columns=True)  # Adjust chunk_size as needed
print(df)

