"""Model definitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F


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
            #nn.BatchNorm1d(out_dim),
            *([nn.Dropout(dropout)] if dropout else []),
            nn.Linear(out_dim, out_dim),
            activation_function,
            #nn.BatchNorm1d(out_dim),
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
                    #nn.BatchNorm1d(block_dim),
                    *([nn.Dropout(dropout)] if dropout else []),
                ))
            prev_dim = block_dim

        self.blocks = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class MDNOutputLayer(nn.Module):
    def __init__(self, hidden_dim, mu, sigma,num_components=2):
        """
        Final layer for a Mixture Density Network.
        
        Args:
            hidden_dim (int): The number of features from your last hidden layer.
            num_components (int): The number of Gaussian clusters (2 for bimodal).
        """
        super(MDNOutputLayer, self).__init__()
        self.num_components = num_components
        
        # A single linear layer that outputs all 3K parameters at once
        # (2 means, 2 standard deviations, 2 mixing weights)
        self.output_layer = nn.Linear(hidden_dim, num_components * 3)


        # Bias layout for K components (output size = 3K):
        #   indices  0  ..  K-1   → pi   (pre-softmax)
        #   indices  K  .. 2K-1   → mu   (direct)
        #   indices 2K  .. 3K-1   → sigma (pre-softplus; softplus(b)+1e-6 = actual sigma)
        with torch.no_grad():
            if mu is not None:
                for k, m in enumerate(mu):
                    self.output_layer.bias[num_components + k] = float(m)
            if sigma is not None:
                for k, s in enumerate(sigma):
                    self.output_layer.bias[2 * num_components + k] = float(s)

    def forward(self, x):
        # Pass the hidden state through the final linear layer
        out = self.output_layer(x)
        
        # Split the output into three equal chunks (size: batch_size x num_components)
        pi, mu, sigma = torch.chunk(out, 3, dim=-1)

        # ---------------------------------------------------------
        # APPLY CONSTRAINTS
        # ---------------------------------------------------------
        # 1. Mixing Weights (pi): Must sum to 1. Use Softmax.
        pi = F.softmax(pi, dim=-1)
        
        # 2. Means (mu): No constraints needed. Leave as is.
        # mu = mu 
        
        # 3. Standard Deviations (sigma): Must be strictly > 0.
        # Use Softplus (a smooth version of ReLU) and add a tiny epsilon 
        # to prevent division-by-zero errors in the loss function.
        sigma = F.softplus(sigma) + 1e-6

        return pi, mu, sigma
    
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=None, activation_function=nn.ReLU(), random_state=42, mu = None, sigma = None, number_of_components=2):
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
        self.mu = mu
        self.sigma = sigma

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

        blocks.append(MDNOutputLayer(prev_dim, mu=self.mu, sigma=self.sigma, num_components=number_of_components))

        self.blocks = nn.ModuleList(blocks)
        #self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    

