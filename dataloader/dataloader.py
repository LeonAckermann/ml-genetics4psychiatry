
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import polars as pl

from dataloader.preprocess import DataConfig, load_csv, load_txt, preprocess, load_txt_polars
from dataloader.GWASDataset import GWASDataset

from dataloader.preprocess import preprocess


def create_significant_rows(illness, distribution, p_value, pval_threshold=0.05):
    data_path = f"./data/sampled_p/{distribution}/sampled_{illness}_p{p_value}.txt"
    df = load_txt_polars(data_path, sep="\t")
    #df_numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    significant_counts_row = (df.iloc[:, 2:] < pval_threshold).sum(axis=1)
    df_sorted = df.copy()
    df_sorted["significant_count"] = significant_counts_row
    df_sorted = df_sorted.sort_values(by="significant_count", ascending=False)

    significant_rows_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}_significant_rows_p{pval_threshold}.txt"
    Path(significant_rows_path).parent.mkdir(parents=True, exist_ok=True)
    df_sorted[["ID", "significant_count"]].to_csv(significant_rows_path, sep="\t", index=False)

def create_significant_columns(illness, distribution, p_value, pval_threshold=0.05):
    data_path = f"./data/sampled_p/{distribution}/sampled_{illness}_p{p_value}.txt"
    df = load_txt_polars(data_path, sep="\t")
    df_pivot = df.T
    #print("Pivoted DataFrame shape:", df_pivot.shape)
    #print("First few rows of pivoted DataFrame:\n", df_pivot.head(2))
    df_pivot = df_pivot.iloc[2:]
    #print("DataFrame shape after removing first two rows:", df_pivot.shape)

    #df_numeric = df_pivot.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    significant_counts_cols = (df_pivot.iloc[:, 1:] < pval_threshold).sum(axis=1)
    #print("Significant counts per column:", significant_counts_cols.shape[0])

    df_pivot["significant_count"] = significant_counts_cols
    #df_pivot = df_pivot.drop(index="Z")
    df_pivot = df_pivot.sort_values(by="significant_count", ascending=False)
    df_pivot = df_pivot.rename_axis("ID").reset_index()
    #df_pivot = df_pivot.rename(columns={df_pivot.columns[0]: "ID"})


    significant_cols_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}_significant_columns_p{pval_threshold}.txt"
    Path(significant_cols_path).parent.mkdir(parents=True, exist_ok=True)
    #df_pivot.reset_index().rename(columns={"index": "feature"}).to_csv(significant_cols_path, sep="\t", index=False)
    df_pivot[["ID", "significant_count"]].to_csv(significant_cols_path, index=False, sep="\t")  

def get_significant(illness, distribution, p_value, row_ratio=0.2, col_ratio=0.1, top_rows=True, top_cols=True, mri_p_value=0.05):
    data_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}.txt"

    # check if data file exists
    significant_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}_significant_rows_p{mri_p_value}.txt"
    if not Path(significant_path).exists():
        # create significant rows file if missing
        create_significant_rows(illness, distribution, p_value, mri_p_value)
    
    significant_cols_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}_significant_columns_p{mri_p_value}.txt"
    if not Path(significant_cols_path).exists():
        # create significant columns file if missing
        create_significant_columns(illness, distribution, p_value, mri_p_value)

    df = load_txt(data_path)
    df_significant_rows = load_txt_polars(significant_path, sep="\t")
    df_significant_cols = load_txt_polars(significant_cols_path, sep="\t")
    #df_significant_cols = df_significant_cols.rename(columns={df_significant_cols.columns[0]: "ID"})


    # Handle legacy significant-columns files that were saved without feature names.
    if "feature" not in df_significant_cols.columns:
        create_significant_columns(illness, distribution, p_value, mri_p_value)
        df_significant_cols = load_txt_polars(significant_cols_path, sep="\t")

    df_significant_rows = df_significant_rows.sort_values(by="significant_count", ascending=False)
    df_significant_cols = df_significant_cols.sort_values(by="significant_count", ascending=False)

    top_rows = int(len(df_significant_rows) * row_ratio)
    top_cols = int(len(df_significant_cols) * col_ratio)

    if top_rows:
        significant_rows = df_significant_rows.head(top_rows)
    else:
        significant_rows = df_significant_rows.tail(len(df_significant_rows) - top_rows)
    
    if top_cols:
        significant_cols = df_significant_cols.head(top_cols)
    else:
        significant_cols = df_significant_cols.tail(len(df_significant_cols) - top_cols)


    significant_cols = significant_cols.reset_index(drop=True)
    significant_cols = significant_cols.iloc[:, 0].tolist()

    df_top = df[df["ID"].isin(significant_rows["ID"])]
    rows_to_keep = ["ID", "Z"] + significant_cols
    df_top = df_top[rows_to_keep]

    df_low = df[~df["ID"].isin(significant_rows["ID"])]
    df_low = df_low[["ID", "Z"] + significant_cols]

    return df_top, df_low

def get_significant_metrics(illness, distribution, p_value, row_ratio=0.2, col_ratio=0.1, top_rows=True, top_cols=True, mri_p_value=0.05):
    data_path = f"./data/sampled_p/{distribution}/sampled_{illness}_p{p_value}.txt"

    # check if data file exists
    significant_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}_significant_rows_p{mri_p_value}.txt"
    if not Path(significant_path).exists():
        # create significant rows file if missing
        create_significant_rows(illness, distribution, p_value, mri_p_value)
    
    significant_cols_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}_significant_columns_p{mri_p_value}.txt"
    if not Path(significant_cols_path).exists():
        # create significant columns file if missing
        create_significant_columns(illness, distribution, p_value, mri_p_value)

    df = load_txt(data_path)
    df_significant_rows = load_txt_polars(significant_path, sep="\t")
    df_significant_cols = load_txt_polars(significant_cols_path, sep="\t")

    df_significant_rows = df_significant_rows.sort_values(by="significant_count", ascending=False)
    df_significant_cols = df_significant_cols.sort_values(by="significant_count", ascending=False)

    top_rows = int(len(df_significant_rows) * row_ratio)
    top_cols = int(len(df_significant_cols) * col_ratio)

    if top_rows:
        significant_rows = df_significant_rows.head(top_rows)
    else:
        significant_rows = df_significant_rows.tail(len(df_significant_rows) - top_rows)
    
    if top_cols:
        significant_cols = df_significant_cols.head(top_cols)
    else:
        significant_cols = df_significant_cols.tail(len(df_significant_cols) - top_cols)

    significant_cols = significant_cols.reset_index(drop=True)
    significant_cols = significant_cols.iloc[:, 0].tolist()

    df_top = df[df["ID"].isin(significant_rows["ID"])]
    rows_to_keep = ["ID", "Z"] + significant_cols
    df = df_top[rows_to_keep]

    # count number of significant features per row
    significant_counts_row = (df.iloc[:, 2:] < mri_p_value).sum(axis=1)
    df_sorted_rows = df.copy()
    df_sorted_rows["significant_count"] = significant_counts_row
    df_sorted_rows["significant_percentage"] = significant_counts_row / (df.shape[1] - 2)
    average_significant_per_row_count = significant_counts_row.mean()
    average_significant_per_row_percentage = (average_significant_per_row_count / (df.shape[1] - 2)).mean()


    # count number of significant samples per feature
    df_pivot = df.T
    df_pivot = df_pivot.iloc[2:]
    significant_counts_cols = (df_pivot.iloc[:, 1:] < mri_p_value).sum(axis=1)
    df_sorted_cols = df_pivot.copy()
    df_sorted_cols["significant_count"] = significant_counts_cols
    df_sorted_cols["significant_percentage"] = significant_counts_cols / (df.shape[0] - 2)
    average_significant_per_col_count = significant_counts_cols.mean()
    average_significant_per_col_percentage = (average_significant_per_col_count / (df.shape[0] - 2)).mean()

    # get overall number of significant entries in the entire dataframe
    total_significant_entries = (df.iloc[:, 2:] < mri_p_value).sum().sum()
    total_significant_percentage = total_significant_entries / ((df.shape[0] - 2) * (df.shape[1] - 2))


    return average_significant_per_row_count, average_significant_per_row_percentage, average_significant_per_col_count, average_significant_per_col_percentage, total_significant_entries, total_significant_percentage

def load_illness_data(illness, in_notebook=True, polars=False, distribution="low", chunk_size=100000, total_chunks=None, p_value="0.001", row_ratio=0.2, col_ratio=0.1, top_rows=True, top_cols=True, mri_p_value=0.05):
    illnesses = {"MDD": "0.001", "ADHD": "0.001", "ASD": "0.001", "OCD": "0.001", "SCZ": "0.0001", "BIP": "0.001", "AZ": "0.001"}

    if row_ratio == 1 and col_ratio == 1:
        if illness not in illnesses:
            raise ValueError(f"Unknown illness: {illness}. Valid options are: {', '.join(illnesses.keys())}")
        pval_threshold = illnesses[illness]
        data_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}.txt"
        if in_notebook:
            data_path = Path("../..") / data_path
        else:
            data_path = Path(data_path).expanduser().resolve()
        print(f"Loading data for illness {illness} at {data_path}"  )
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found for illness {illness} at {data_path}")

        if polars:
            df_illness = load_txt_polars(Path(data_path), chunk_size=chunk_size, total_chunks=total_chunks)
        else:
            df_illness = load_txt(Path(data_path), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_illness, _ = get_significant(illness, distribution, p_value, row_ratio=row_ratio, col_ratio=col_ratio, top_rows=top_rows, top_cols=top_cols, mri_p_value=mri_p_value)

    return df_illness

def prepare_data_splits(df, testsize, illness, nsplits, save=True):
    """
    Input: DataFrame, target column name, test size, illness name, number of splits, whether to save splits
    Output: Saves train/test splits as .pt files in data/splits/{illness}_{nsplits}/seed_{seed}/
    """
    seeds = [42 + i for i in range(nsplits)]
    target = f"Z"
    
    for seed in seeds:
        X_train, y_train, X_test, y_test = preprocess(df, target, testsize, seed)
        output_dir = Path(f"./data/splits/{illness}_{nsplits}/seed_{seed}").expanduser().resolve()
        if seed == 42:
            print(f"saved splits for seed {seed} at {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if save:
            import numpy as np
            np.savez(output_dir / f"train_split_{seed}.npz", X=X_train, y=y_train)
            np.savez(output_dir / f"test_split_{seed}.npz", X=X_test, y=y_test)

def load_data_split(illness, nsplits, seed):
    output_dir = Path(f"./data/splits/{illness}_{nsplits}/seed_{seed}").expanduser().resolve()
    train_path = output_dir / f"train_split_{seed}.npz"
    test_path = output_dir / f"test_split_{seed}.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Data splits not found for seed {seed} at {output_dir}")
    import numpy as np
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    return train_data["X"], train_data["y"], test_data["X"], test_data["y"]
