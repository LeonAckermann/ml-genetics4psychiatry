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
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=None, activation_function=nn.ReLU()):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_function)
            layers.append(nn.BatchNorm1d(h_dim))

            if dropout:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class ResidualBlock(nn.Module):
    """A block of two hidden layers with a skip connection."""
    def __init__(self, in_dim, out_dim, dropout=None, activation_function=nn.ReLU()):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation_function,
            nn.BatchNorm1d(out_dim),
            *([nn.Dropout(dropout)] if dropout else []),
            nn.Linear(out_dim, out_dim),
            activation_function,
            nn.BatchNorm1d(out_dim),
            *([nn.Dropout(dropout)] if dropout else []),
        )
        # projection if dimensions change
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class ResidualDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=None, activation_function=nn.ReLU(), random_state=42):
        super().__init__()

        # set random seed for reproducibility
        torch.manual_seed(random_state)
        # also for cuda if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        # set for mps
        #if torch.backends.mps.is_available():
        #    torch.backends.mps.manual_seed_all(random_state)

        blocks = []
        prev_dim = input_dim

        # Group hidden dims in pairs for residual blocks
        for i in range(0, len(hidden_dims), 2):
            block_dim = hidden_dims[i]
            if i + 1 < len(hidden_dims):
                # Two-layer residual block (both layers use hidden_dims[i] width)
                blocks.append(ResidualBlock(prev_dim, block_dim, dropout, activation_function))
            else:
                # Odd layer out — single layer without skip connection
                blocks.append(nn.Sequential(
                    nn.Linear(prev_dim, block_dim),
                    activation_function,
                    nn.BatchNorm1d(block_dim),
                    *([nn.Dropout(dropout)] if dropout else []),
                ))
            prev_dim = block_dim

        self.blocks = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)
    
    

