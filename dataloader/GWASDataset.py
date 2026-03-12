import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

class GWASDataset(Dataset):
    def __init__(self, features, targets):
        x = features.to_numpy() if hasattr(features, "to_numpy") else np.asarray(features)
        y = targets.to_numpy() if hasattr(targets, "to_numpy") else np.asarray(targets)

        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]