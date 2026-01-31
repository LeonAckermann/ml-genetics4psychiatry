"""Model definitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class BaselineModel:
    """Simple baseline that predicts the mean of the target."""

    target_column: str

    def fit(self, df: pd.DataFrame) -> "BaselineModel":
        if self.target_column not in df.columns:
            raise ValueError(f"Target column not found: {self.target_column}")
        self.mean_ = float(df[self.target_column].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Model must be fitted before prediction.")
        return np.full(shape=(len(df),), fill_value=self.mean_, dtype=float)
    
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=None):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())

            if dropout:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    

