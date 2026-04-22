from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn.finetuning import FinetunedTabPFNRegressor
from tabpfn.constants import ModelVersion

from dataloader import load_txt, preprocess, GWASDataset

from pathlib import Path
import sys
import torch
import torch


class TabPFNModel:
    def __init__(self, random_state=42):
        self.model = TabPFNRegressor()  # Uses TabPFN-2.5 weights, trained on synthetic data only.
        # To use TabPFN v2:
        # Create final pipeline with best parameters

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    

class FinetunedTabPFNModel:
    def __init__(self, random_state=42, device="cuda", epochs=30, learning_rate=1e-5):
        self.model = FinetunedTabPFNRegressor(device=device, epochs=epochs, learning_rate=float(learning_rate), random_state=random_state)  # Uses TabPFN-2.5 weights, trained on synthetic data only.
        # To use TabPFN v2:
        # Create final pipeline with best parameters

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
