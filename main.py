"""Entry point for running experiments from a YAML config file.

Usage:
    python main.py --config experiments/linear_regression_scz.yaml
"""

from __future__ import annotations

import argparse
import subprocess

from pathlib import Path

import json
from datetime import datetime

from itertools import product
import yaml
import optuna
from sklearn.model_selection import KFold, cross_val_score
from dataloader.pipeline import aligne_illness_mri, call_plink2, aligne_clumped_illness_mri, construct_gwas_mri
from dataloader.preprocess import sample
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from dataloader import load_illness_data, prepare_data_splits, load_data_split
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from optuna.samplers import TPESampler


# ---------------------------------------------------------------------------
# Data factory
# ---------------------------------------------------------------------------


def pipeline(cfg: dict) -> None:
    """Run data processing pipeline steps based on config."""
    # Example: if cfg["data"]["align_illness_mri"]:
    #              aligne_illness_mri(cfg["data"]["illness"])
    
    


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
    elif name == "tabpfn":
        from model import TabPFNModel
        if cfg["model"].get("finetune", False):
            from model import FinetunedTabPFNModel
            return FinetunedTabPFNModel(random_state=cfg.get("seed", 42), device=cfg["model"].get("device", "cuda"), epochs=cfg["model"].get("epochs", 30), learning_rate=cfg["model"].get("learning_rate", 1e-5))
        return TabPFNModel(random_state=cfg.get("seed", 42))

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
            "patience": model_cfg.get("patience", 20),
        }
    
        

    elif name == "baseline":
        from model import BaselineModel
        return BaselineModel(target_column=cfg["data"]["target"])

    else:
        raise ValueError(f"Unknown model: {name}")


def nested_cv_xgboost(X, y, outer_cv=5, inner_cv=3, n_trials=50, search_space=None):
    """Perform nested cross-validation using Optuna for XGBoost."""
    from xgboost import XGBRegressor

    if search_space is None:
        search_space = {
            'n_estimators': [100, 1000],
            'max_depth': [3, 10],
            'learning_rate': [0.01, 0.3],
            'subsample': [0.6, 1.0],
            'colsample_bytree': [0.6, 1.0],
            'reg_alpha': [0, 1.0],
            'reg_lambda': [0, 5.0],
            'min_child_weight': [1, 10],
            'gamma': [0, 1.0],
        }

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    outer_scores = []

    optuna.logging.set_verbosity(optuna.logging.INFO)

    fold_best_params = []
    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X_arr)):
        print(f"--- Starting Outer Fold {fold + 1}/{outer_cv} ---")
        X_train_outer, X_test_outer = X_arr[train_idx], X_arr[test_idx]
        y_train_outer, y_test_outer = y_arr[train_idx], y_arr[test_idx]

        def objective(trial):
            params = {
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': 1,
            }
            for param_name, bounds in search_space.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    low, high = bounds
                    if param_name in ['n_estimators', 'max_depth']:
                        params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                    else:
                        log = param_name == 'learning_rate'
                        params[param_name] = trial.suggest_float(param_name, float(low), float(high), log=log)
                elif isinstance(bounds, list):
                    params[param_name] = trial.suggest_categorical(param_name, bounds)
                else:
                    params[param_name] = bounds

            print(f"    Inner trial {trial.number + 1}/{n_trials} evaluating params: {params}")
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X_train_outer, y_train_outer, cv=inner_cv, scoring='r2', n_jobs=1)
            mean_score = scores.mean()
            print(f"    Trial {trial.number + 1} mean inner R2: {mean_score:.4f}")
            return mean_score
            
        sampler = TPESampler(seed=42 + fold)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        print(f"  Completed inner optimization for fold {fold + 1}, best inner R2: {study.best_value:.4f}")
        best_params = study.best_params
        best_params.update({'random_state': 42, 'verbosity': 0, 'n_jobs': 1})
        fold_best_params.append(best_params)
        best_model_for_fold = XGBRegressor(**best_params)
        best_model_for_fold.fit(X_train_outer, y_train_outer)

        y_pred = best_model_for_fold.predict(X_test_outer)
        score = r2_score(y_test_outer, y_pred)
        outer_scores.append(score)

        print(f"Outer Fold {fold + 1} R2 Score: {score:.4f}")
        print(f"Best Params for Fold {fold + 1}: {best_params}\n")

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    print("=== Final Nested CV Results ===")
    print(f"Average R2: {mean_score:.4f} (+/- {std_score:.4f})")

    return best_model_for_fold, outer_scores, mean_score, std_score, fold_best_params


def nested_cv_dnn(X, y, model_name='residual_dnn', outer_cv=5, inner_cv=3, n_trials=50, search_space=None, val_size=0.1):
    """Perform nested cross-validation using Optuna for DNN or ResidualDNN."""
    from model import DNN, ResidualDNN

    if model_name == 'dnn':
        model_class = DNN
    elif model_name == 'residual_dnn':
        model_class = ResidualDNN
    else:
        raise ValueError(f"Unsupported DNN model for search: {model_name}")

    if search_space is None:
        search_space = {
            'hidden_dim': [32, 128],
            'n_layers': [1, 4],
            'dropout': [0.0, 0.5],
            'learning_rate': [1e-4, 1e-2],
            'weight_decay': [0.0, 0.001],
            'batch_size': [16, 64],
            'epochs': [20, 60],
            'patience': [5, 30],
        }

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    outer_scores = []
    fold_best_params = []

    optuna.logging.set_verbosity(optuna.logging.INFO)

    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X_arr)):
        print(f"--- Starting Outer Fold {fold + 1}/{outer_cv} ---")
        X_train_outer, X_test_outer = X_arr[train_idx], X_arr[test_idx]
        y_train_outer, y_test_outer = y_arr[train_idx], y_arr[test_idx]

        def objective(trial):
            trial_params = {}
            for param_name, bounds in search_space.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    low, high = bounds
                    if param_name in ['hidden_dim', 'n_layers', 'batch_size', 'epochs', 'patience']:
                        trial_params[param_name] = int(trial.suggest_int(param_name, int(low), int(high)))
                    else:
                        log = param_name in ['learning_rate']
                        trial_params[param_name] = float(trial.suggest_float(param_name, float(low), float(high), log=log))
                elif isinstance(bounds, list):
                    trial_params[param_name] = trial.suggest_categorical(param_name, bounds)
                else:
                    trial_params[param_name] = bounds

            hidden_dims = [trial_params['hidden_dim']] * trial_params['n_layers']
            model_cfg = {
                'class': model_class,
                'hidden_dims': hidden_dims,
                'output_dim': 1,
                'dropout': trial_params.get('dropout', None),
                'lr': trial_params.get('learning_rate', 1e-3),
                'epochs': trial_params.get('epochs', 30),
                'batch_size': trial_params.get('batch_size', 32),
                'patience': trial_params.get('patience', 20),
                'seed': 42,
            }
            optimizer_cfg = {
                'weight_decay': trial_params.get('weight_decay', 0.0),
            }

            print(f"    Trial {trial.number + 1}/{n_trials} params: {trial_params}")
            inner_scores = []
            inner_kfold = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(X_train_outer)):
                X_train_inner = X_train_outer[inner_train_idx]
                y_train_inner = y_train_outer[inner_train_idx]
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                    X_train_inner,
                    y_train_inner,
                    test_size=val_size,
                    random_state=42,
                )
                X_test_inner = X_train_outer[inner_val_idx]
                y_test_inner = y_train_outer[inner_val_idx]

                # Clone config for each inner fold
                inner_model_cfg = model_cfg.copy()
                preds = train_dnn(
                    inner_model_cfg,
                    X_train=X_train_inner,
                    y_train=y_train_inner,
                    X_val=X_val_inner,
                    y_val=y_val_inner,
                    X_test=X_test_inner,
                    y_test=y_test_inner,
                    cfg={'verbose': False},
                )
                val_score = r2_score(y_test_inner, preds)
                inner_scores.append(val_score)
                print(f"        Inner fold {inner_fold + 1} R2: {val_score:.4f}")

            mean_inner_score = float(np.mean(inner_scores))
            print(f"    Trial {trial.number + 1} mean inner R2: {mean_inner_score:.4f}")
            return mean_inner_score

        sampler = TPESampler(seed=42 + fold)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        fold_best_params.append(best_params)
        print(f"  Completed inner optimization for fold {fold + 1}, best inner R2: {study.best_value:.4f}")

        best_hidden_dims = [int(best_params['hidden_dim'])] * int(best_params['n_layers'])
        final_model_cfg = {
            'class': model_class,
            'hidden_dims': best_hidden_dims,
            'output_dim': 1,
            'dropout': best_params.get('dropout', None),
            'lr': best_params.get('learning_rate', 1e-3),
            'epochs': int(best_params.get('epochs', 30)),
            'batch_size': int(best_params.get('batch_size', 32)),
            'patience': int(best_params.get('patience', 20)),
            'seed': 42,
        }

        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=val_size,
            random_state=42,
        )
        print(f"  Training final DNN on outer training set with {len(X_train_final)} train / {len(X_val_final)} val samples")

        final_preds = train_dnn(
            final_model_cfg,
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            X_test_outer,
            y_test_outer,
            {'model': final_model_cfg},
        )
        score = r2_score(y_test_outer, final_preds)
        outer_scores.append(score)
        print(f"Outer Fold {fold + 1} R2 Score: {score:.4f}")
        print(f"Best Params for Fold {fold + 1}: {best_params}\n")

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    print("=== Final Nested CV Results ===")
    print(f"Average R2: {mean_score:.4f} (+/- {std_score:.4f})")

    return outer_scores, mean_score, std_score, fold_best_params


def nested_cv_regularized_regression(X, y, model_name='lasso_regression', outer_cv=5, inner_cv=3, n_trials=50, search_space=None):
    """Perform nested cross-validation using Optuna for Lasso or Ridge regression."""
    if model_name == 'lasso_regression':
        from model import LassoRegressionModel
        model_class = LassoRegressionModel
        default_alpha_range = [1e-4, 10.0]
    elif model_name == 'ridge_regression':
        from model import RidgeRegressionModel
        model_class = RidgeRegressionModel
        default_alpha_range = [1e-4, 10000.0]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if search_space is None:
        search_space = {
            'alpha': default_alpha_range,
        }

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    outer_scores = []
    fold_best_params = []

    optuna.logging.set_verbosity(optuna.logging.INFO)

    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X_arr)):
        print(f"--- Starting Outer Fold {fold + 1}/{outer_cv} ---")
        X_train_outer, X_test_outer = X_arr[train_idx], X_arr[test_idx]
        y_train_outer, y_test_outer = y_arr[train_idx], y_arr[test_idx]

        def objective(trial):
            alpha = trial.suggest_float('alpha', search_space['alpha'][0], search_space['alpha'][1], log=True)
            model = model_class(best_alpha=alpha)
            scores = cross_val_score(model, X_train_outer, y_train_outer, cv=inner_cv, scoring='r2')
            mean_score = scores.mean()
            print(f"    Trial {trial.number + 1}/{n_trials} alpha={alpha:.6f}, mean inner R2: {mean_score:.4f}")
            return mean_score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        print(f"  Completed inner optimization for fold {fold + 1}, best inner R2: {study.best_value:.4f}")
        best_params = {'alpha': study.best_params['alpha']}
        fold_best_params.append(best_params)
        
        best_model = model_class(best_alpha=best_params['alpha'])
        best_model.fit(X_train_outer, y_train_outer)
        y_pred = best_model.predict(X_test_outer)
        score = r2_score(y_test_outer, y_pred)
        outer_scores.append(score)

        print(f"Outer Fold {fold + 1} R2 Score: {score:.4f}")
        print(f"Best Params for Fold {fold + 1}: {best_params}\n")

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    print("=== Final Nested CV Results ===")
    print(f"Average R2: {mean_score:.4f} (+/- {std_score:.4f})")

    return outer_scores, mean_score, std_score, fold_best_params


def normal_cv_linear_regression(X, y, cv=5):
    """Perform normal cross-validation for Linear regression (no HPO)."""
    from model import LinearRegressionModel

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    
    model = LinearRegressionModel()
    scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring='r2')
    
    mean_score = scores.mean()
    std_score = scores.std()
    
    print("=== Linear Regression CV Results ===")
    print(f"Fold Scores: {scores}")
    print(f"Average R2: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return scores, mean_score, std_score


def search_hyperparams(model_name, X, y, n_trials=100, outer_cv=5, inner_cv=3, search_space=None):
    """Perform hyperparameter optimization using Optuna for the specified model."""
    if model_name == "xgboost":
        _, outer_scores, mean_score, std_score, fold_best_params = nested_cv_xgboost(
            X,
            y,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            n_trials=n_trials,
            search_space=search_space,
        )
        return {
            'outer_scores': outer_scores,
            'mean_r2': mean_score,
            'std_r2': std_score,
            'fold_best_params': fold_best_params,
            'search_space': search_space,
        }
    elif model_name in ["dnn", "residual_dnn"]:
        outer_scores, mean_score, std_score, fold_best_params = nested_cv_dnn(
            X,
            y,
            model_name=model_name,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            n_trials=n_trials,
            search_space=search_space,
            val_size=0.1,
        )
        return {
            'outer_scores': outer_scores,
            'mean_r2': mean_score,
            'std_r2': std_score,
            'fold_best_params': fold_best_params,
            'search_space': search_space,
        }
    elif model_name in ["lasso_regression", "ridge_regression"]:
        outer_scores, mean_score, std_score, fold_best_params = nested_cv_regularized_regression(
            X,
            y,
            model_name=model_name,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            n_trials=n_trials,
            search_space=search_space,
        )
        return {
            'outer_scores': outer_scores,
            'mean_r2': mean_score,
            'std_r2': std_score,
            'fold_best_params': fold_best_params,
            'search_space': search_space,
        }
    elif model_name == "linear_regression":
        scores, mean_score, std_score = normal_cv_linear_regression(
            X,
            y,
            cv=outer_cv,
        )
        return {
            'cv_scores': scores.tolist(),
            'mean_r2': float(mean_score),
            'std_r2': float(std_score),
        }
    else:
        raise ValueError(f"Hyperparameter search not implemented for model: {model_name}")


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_sklearn(model, X_train, y_train, X_test, y_test, cfg):
    """Train an sklearn-style model (fit/predict interface)."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


def train_dnn(model_cfg, X_train, y_train, X_val, y_val, X_test, y_test, cfg=None):
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
    train_loader = DataLoader(GWASDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GWASDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(GWASDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"])

    # learning rate scheduler
    #T_0 = 15
    #T_mult = 2
    #eta_min = 1e-6
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    # use a simple StepLR scheduler that decays the learning rate by 0.1 every 30 epochs
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_losses = []
    test_losses = []
    learning_rates = []

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # model to device
    model.to(device)

    best_model = None
    best_val_loss = [float("inf"), -1]  # [loss, epoch]
    epochs_since_improvement = 0
    patience = int(model_cfg.get("patience", 20))

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
        #scheduler.step()

        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if cfg and cfg.get("verbose", True):
                print(f"  Epoch {epoch+1}/{model_cfg['epochs']}, Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_preds = model(val_X)
                loss_value = criterion(val_preds, val_y).item()
                val_loss_total += loss_value * val_X.size(0)

        average_val_loss = val_loss_total / len(val_loader.dataset)
        test_losses.append(average_val_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if cfg and cfg.get("verbose", True):
                print(f"    Validation Loss: {average_val_loss:.4f}")

        if average_val_loss < best_val_loss[0]:
            best_val_loss = [average_val_loss, epoch]
            best_model = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            if cfg and cfg.get("verbose", True):
                print(f"Early stopping triggered after {patience} epochs without validation loss improvement.")
            break

    
    # Load best model weights
    model.load_state_dict(best_model)
    model.eval()
    if cfg and cfg.get("verbose", True):
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
    if "stderr" in metric_names:
        stderr = np.sqrt(mean_squared_error(y_test, preds) / len(y_test))
        results["stderr"] = float(stderr)
        print(f"  StdErr:      {results['stderr']:.4f}")

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

    plink_cfg = cfg["plink2"]
    construct_cfg = cfg.get("construct_gwas_mri", {})
    data_cfg = cfg["data"]
    illness = data_cfg["illness"]
    n_splits = data_cfg.get("n_splits", 5)
    test_size = data_cfg.get("test_size", 0.2)
    save_splits = data_cfg.get("save_splits", True)
    total_chunks = plink_cfg.get("total_chunks", None)
    chunk_size = plink_cfg.get("chunk_size", 10000)

    results_file = None
    output = {}
    written = False


    # Construct merged GWAS MRI file (one-time preprocessing step)
    if construct_cfg.get("run", False):
        print("\nConstructing merged GWAS MRI file...")
        stats = construct_gwas_mri(
            path=construct_cfg["input_path"],
            output_path=construct_cfg["output_path"],
            chunk_size=construct_cfg.get("chunk_size", 10000),
            total_chunks=construct_cfg.get("total_chunks", None),
            polars=construct_cfg.get("polars", False)
        )
        print("Done constructing GWAS MRI file.")
        output["gwas_mri_stats"] = stats

    # Run data processing pipeline
    if plink_cfg.get("prepare", False):

        mri_path = plink_cfg.get("mri", None)
        print("\nRunning data processing pipeline...")
        
        first_alignment = aligne_illness_mri(illness, verbose=True, chunk_size=chunk_size, total_chunks=total_chunks, mri_path=mri_path, polars=plink_cfg.get("polars", False))
        #subprocess.run("ln -sf $HOME/tools/plink2/plink2 $HOME/tools/bin/plink2", shell=True)
        
        # run this command with subprocess echo 'export PATH="$HOME/tools/bin:$PATH"' >> ~/.bashrc
        #subprocess.run("echo 'export PATH=\"$HOME/tools/bin:$PATH\"' >> ~/.bashrc", shell=True)
        #subprocess.run("source ~/.bashrc", shell=True)
        plink2 = {
            "--bfile": plink_cfg["ref"],
            "--clump": plink_cfg["aligned"],
            "--clump-kb": plink_cfg["clump_kb"],
            "--clump-r2": float(plink_cfg["r2"]),
            "--clump-p1": plink_cfg["p_clump"],
            "--clump-p2": plink_cfg["p_clump"],
            "--out": plink_cfg["output"],
        }
        call_plink2(plink2)
        second_alignment = aligne_clumped_illness_mri(illness, verbose=True, polars=plink_cfg.get("polars", False), mri_path=mri_path, chunk_size=chunk_size, total_chunks=total_chunks)
        output["illness_mri_alignment"] = first_alignment
        output["plink2"] = plink2
        output["clumped_illness_mri_alignment"] = second_alignment

    #else:
    #    # check if aligned data exists, if not run the alignment step
    #    aligned_path = Path(f"./data/pipeline/final/aligned_clumped_{illness}.txt")
    #    if not aligned_path.exists():
    #        # tell uer to run the pipeline first
    #        print(f"Aligned data not found at {aligned_path}. Please run the data processing pipeline first with --prepare flag.")
    #        return
        
    # Hyperparameter search block only runs when HPO is enabled in the experiment YAML.
    hpo_enabled = False
    hpo_cfg = {}
    if cfg.get("hpo") is True:
        hpo_enabled = True
    elif isinstance(cfg.get("hpo"), dict):
        hpo_cfg = cfg["hpo"]
        if hpo_cfg.get("run", True) is not False:
            hpo_enabled = True

    if hpo_enabled:
        print("HPO configuration detected. Nested HPO search will run during each experiment.")

    if cfg["experiment"].get("run", True) or hpo_enabled:

    
        # construct data dict based on data_cfg distribution, p_clump, and illness, this could be multiple runs
        for dist, p, illness in product(data_cfg.get("distribution", []), data_cfg.get("p_clump", []), data_cfg.get("illness", [])):
            
            print("Starting experiment with illness={}, p_clump={}, distribution={}...".format(illness, p, dist))
            experiment_name = f"{cfg['model']['name']}_{illness}_p{p}_{dist}"
            results_dir = Path("./results") / experiment_name
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"{experiment_name}_{timestamp}.json"
            output = {}
            
            if data_cfg.get("sampling", False):
                print(f"\nSampling data for illness={illness}, p_clump={p}, distribution={dist}...")
                metrics = sample(
                    p_value=p,
                    distribution=dist,
                    illness=illness,
                    polars=data_cfg.get("polars", False),
                    chunk_size=data_cfg.get("chunk_size", 100000),
                    total_chunks=data_cfg.get("total_chunks", None)
                )
                output[f"sampling_metrics_{illness}_{dist}_p{p}"] = metrics
            else:
                print(f"\nSkipping sampling for illness={illness}, p_clump={p}, distribution={dist} since sampling flag is set to False.")
                # check whether data exists, if not raise error
                data_path = Path(f"./data/sampled/{dist}/sampled_{illness}_p{p}.txt")
                if not data_path.exists():
                    raise FileNotFoundError(f"Sampled data not found at {data_path}. Please run the sampling step first with sampling flag set to True.")

            # Load data and prepare splits, 
            df = load_illness_data(illness, in_notebook=False, polars=data_cfg.get("polars", True), distribution=dist, chunk_size=chunk_size, total_chunks=total_chunks, p_value=p)
            print(f"Loaded data for {illness}: {df.shape[0]} samples, {df.shape[1]} features")

            hpo_results = None
            if hpo_enabled and cfg["model"]["name"] in ["xgboost", "dnn", "residual_dnn", "lasso_regression", "ridge_regression", "linear_regression"]:
                print(f"Running nested CV HPO for experiment {experiment_name}...")
                df_pandas = df.to_pandas() if hasattr(df, "to_pandas") else df
                id_cols = [col for col in ["ID"] if col in df_pandas.columns]
                X = df_pandas.drop(columns=[data_cfg["target"]] + id_cols)
                y = df_pandas[data_cfg["target"]]
                hpo_results = search_hyperparams(
                    cfg["model"]["name"],
                    X,
                    y,
                    n_trials=hpo_cfg.get("n_trials", 100),
                    outer_cv=hpo_cfg.get("outer_cv", 5),
                    inner_cv=hpo_cfg.get("inner_cv", 3),
                    search_space=hpo_cfg.get("search_space"),
                )
                output["hpo"] = hpo_results

            if cfg["experiment"].get("run", True):
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
                    elif cfg["model"]["name"] in ["linear_regression", "ridge_regression", "lasso_regression", "xgboost", "elastic_regression", "tabpfn"]:
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

                output["experiment"] = experiment_name
                output["config"] = cfg
                output["timestamp"] = timestamp
                output["per_seed"] = seed_results
                output["aggregated"] = aggregated_metrics

                with open(results_file, "w") as f:
                    json.dump(output, f, indent=2)
                    written = True

                print(f"\nResults saved to {results_file}")
            else:
                output["experiment"] = experiment_name
                output["config"] = cfg
                output["timestamp"] = timestamp
                with open(results_file, "w") as f:
                    json.dump(output, f, indent=2)
                    written = True

                print(f"\nHPO-only results saved to {results_file}")

    

        
    else:
        print("Experiment run flag is set to False. Skipping training and evaluation.")

    if not written and results_file is not None:
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {results_file}")
    elif results_file is None:
        print("No results file was created because no experiment or HPO run was enabled.")


if __name__ == "__main__":
    main()
