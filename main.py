"""Entry point for running experiments from a YAML config file.

Usage:
    python main.py --config experiments/linear_regression_scz.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json
from datetime import datetime

import yaml
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from dataloader import load_illness_data, prepare_data_splits, load_data_split
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader



# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict):
    """Instantiate a model from the config dict."""
    name = cfg["model"]["name"]

    if name == "linear_regression":
        from model import LinearRegressionModel
        return LinearRegressionModel()
    
    elif name == "ridge_regression":
        from model import RidgeRegressionModel
        return RidgeRegressionModel(cfg["model"].get("best_alpha", 1000.0)) # hyperparameter from previous tuning in notebook
    
    elif name == "lasso_regression":
        from model import LassoRegressionModel
        return LassoRegressionModel(cfg["model"].get("best_alpha", 0.02123011124333675)) # hyperparameter from previous tuning in notebook
    elif name == "elastic_regression":
        from model import ElasticRegressionModel
        return ElasticRegressionModel(
            best_l1_ratio=cfg["model"].get("best_l1_ratio", 0.1), 
            best_alpha=cfg["model"].get("best_alpha", 0.06951927961775606)
        ) # hyperparameters from previous tuning in notebook

    elif name == "xgboost":
        from model import XGBoostTreeModel
        return XGBoostTreeModel(random_state=cfg.get("seed", 42), n_estimators=cfg["model"].get("n_estimators", 100), max_depth=cfg["model"].get("max_depth", 6), learning_rate=cfg["model"].get("learning_rate", 0.1), subsample=cfg["model"].get("subsample", 0.8), colsample_bytree=cfg["model"].get("colsample_bytree", 0.8))

    elif name == "dnn" or name == "residual_dnn":
        import torch
        from torch import nn, optim
        from model import DNN, ResidualDNN
        model_cfg = cfg["model"]
        if name == "dnn":
            model_class = DNN
        elif name == "residual_dnn":
            model_class = ResidualDNN
        # input_dim is set later after loading data
        return {
            "class": model_class,
            "hidden_dims": model_cfg.get("hidden_dims", [50, 50]),
            "output_dim": model_cfg.get("output_dim", 1),
            "dropout": model_cfg.get("dropout", None),
            "lr": model_cfg.get("learning_rate", 0.001),
            "epochs": model_cfg.get("epochs", 100),
            "batch_size": model_cfg.get("batch_size", 32),
        }
        

    elif name == "baseline":
        from model import BaselineModel
        return BaselineModel(target_column=cfg["data"]["target"])

    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_sklearn(model, X_train, y_train, X_test, y_test, cfg):
    """Train an sklearn-style model (fit/predict interface)."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


def train_dnn(model_cfg, X_train, y_train, X_val, y_val, X_test, y_test, cfg):
    """Train a DNN model with PyTorch."""
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from dataloader import GWASDataset

    input_dim = X_train.shape[1]
    model = model_cfg["class"](
        input_dim=input_dim,
        hidden_dims=model_cfg["hidden_dims"],
        output_dim=model_cfg["output_dim"],
        dropout=model_cfg["dropout"],
        random_state=model_cfg.get("seed", 42),
    )
    

    batch_size = model_cfg["batch_size"]
    train_loader = DataLoader(GWASDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GWASDataset(X_val, y_val), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(GWASDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"])

    # learning rate scheduler
    T_0 = 15
    T_mult = 2
    eta_min = 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    train_losses = []
    test_losses = []
    learning_rates = []

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # model to device
    model.to(device)

    best_model = None
    best_val_loss = [float("inf"), -1]  # [loss, epoch]
    val_higher = 0
    for epoch in range(model_cfg["epochs"]):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Move to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step()
        
        train_losses.append(total_loss / len(train_loader.dataset))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{model_cfg['epochs']}, Loss: {avg_loss:.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_preds = model(val_X)
                val_loss = criterion(val_preds, val_y).item()
                test_loss += val_loss * val_X.size(0)
           
        average_test_loss = test_loss / len(val_loader.dataset)
        test_losses.append(average_test_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Validation Loss: {average_test_loss:.4f}")   

        # save best model based on validation loss
        if average_test_loss < best_val_loss[0]:
            best_val_loss = [average_test_loss, epoch]
            best_model = model.state_dict()

        if epoch > 0 and test_losses[-1] > test_losses[-2]:
            val_higher += 1
        else:        
            val_higher = 0

        if val_higher >= 20:
            print("Early stopping triggered due to 20 consecutive epochs with higher validation loss than training loss.")
            break

    
    # Load best model weights
    model.load_state_dict(best_model)
    model.eval()
    print(f"\nBest validation loss: {best_val_loss[0]:.4f} at epoch {best_val_loss[1]+1}")
    print("Evaluating on test set with best model...")
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds


def evaluate(y_test, preds, cfg):
    """Compute, print, and return evaluation metrics as a dict."""
    eval_cfg = cfg.get("evaluation", {})
    metric_names = eval_cfg.get("metrics", ["r2", "mse"])
    results = {}

    print("\n" + "=" * 44)
    print("             Evaluation Results")
    print("=" * 44)

    if "r2" in metric_names:
        results["r2"] = float(r2_score(y_test, preds))
        print(f"  R² Score:    {results['r2']:.4f}")
    if "mse" in metric_names:
        results["mse"] = float(mean_squared_error(y_test, preds))
        print(f"  MSE:         {results['mse']:.4f}")

    print("=" * 44)

    # Optional binary classification metrics
    threshold = eval_cfg.get("binary_threshold")
    if threshold is not None:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report,
        )
        y_true_bin = (np.array(y_test) >= threshold).astype(int)
        y_pred_bin = (np.array(preds) >= threshold).astype(int)

        results["binary_threshold"] = threshold
        results["accuracy"] = float(accuracy_score(y_true_bin, y_pred_bin))
        results["precision"] = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        results["recall"] = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        results["f1"] = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

        print(f"\n  Binary classification (threshold = {threshold}):")
        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  Precision:   {results['precision']:.4f}")
        print(f"  Recall:      {results['recall']:.4f}")
        print(f"  F1 Score:    {results['f1']:.4f}")
        print("\n" + classification_report(
            y_true_bin, y_pred_bin,
            target_names=[f"< {threshold}", f">= {threshold}"],
            zero_division=0,
        ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML genetics experiments from a YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    illness = data_cfg["illness"]
    n_splits = data_cfg.get("n_splits", 5)
    test_size = data_cfg.get("test_size", 0.2)
    save_splits = data_cfg.get("save_splits", True)

    print(f"Experiment config: {config_path.name}")
    print(f"  Illness:  {illness}")
    print(f"  Model:    {cfg['model']['name']}")
    print(f"  Splits:   {n_splits}")
    print(f"  Test size: {test_size}")
    print(f"  Save splits: {save_splits}")

    # Load data and prepare splits
    df = load_illness_data(illness, in_notebook=False)
    print(f"Loaded data for {illness}: {df.shape[0]} samples, {df.shape[1]} features")

    if save_splits:
        prepare_data_splits(df, test_size, illness, nsplits=n_splits, save=True)

    # Build model
    model = build_model(cfg)

    # Run over all splits
    all_preds, all_y_test = [], []
    seeds = [42 + i for i in range(n_splits)]
    seed_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Split {i+1}/{n_splits} (seed={seed}) ---")
        cfg["seed"] = seed  # Add seed to config for reproducibility
        X_train, y_train, X_test, y_test = load_data_split(illness, n_splits, seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed+1)
        

        # Convert tensors to numpy if needed
        #if hasattr(X_train, "numpy"):
        #    X_train, y_train = X_train.numpy(), y_train.numpy()
        #    X_val, y_val = X_val.numpy(), y_val.numpy()
        #    X_test, y_test = X_test.numpy(), y_test.numpy()

        if cfg["model"]["name"] in ["dnn", "residual_dnn"]:
            preds = train_dnn(model, X_train, y_train, X_val, y_val, X_test, y_test, cfg)
        elif cfg["model"]["name"] in ["linear_regression", "ridge_regression", "lasso_regression", "xgboost", "elastic_regression"]:
            preds = train_sklearn(model, X_train, y_train, X_test, y_test, cfg)

        split_metrics = evaluate(y_test, preds, cfg)
        split_metrics["seed"] = seed
        seed_results.append(split_metrics)
        all_preds.append(preds)
        all_y_test.append(y_test)

    # Aggregate results across splits
    all_preds = np.concatenate(all_preds)
    all_y_test = np.concatenate(all_y_test)
    print("\n###  Aggregated results across all splits  ###")
    aggregated_metrics = evaluate(all_y_test, all_preds, cfg)

    # Save results to JSON
    experiment_name = config_path.stem
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{experiment_name}_{timestamp}.json"

    output = {
        "experiment": experiment_name,
        "config": cfg,
        "timestamp": timestamp,
        "per_seed": seed_results,
        "aggregated": aggregated_metrics,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
