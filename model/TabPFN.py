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


def _pin_kv_cache_to_device(model) -> None:
    """Keep the KV cache on the compute device after .fit(), where supported.

    tabpfn_extensions warns during SHAP that ``executor_.keep_cache_on_device``
    is False and should be set after fit. On tabpfn 7.0.1 that flag does not
    exist: ``InferenceEngineCacheKV`` holds the cache in CPU RAM by design and
    reads no such attribute (zero references anywhere in the tabpfn package).
    Assigning it there would silence the warning without changing anything, so
    this only sets the flag on versions that actually define it — the warning
    stays visible otherwise, which is the honest signal.

    Only relevant with ``fit_mode="fit_with_cache"``, which src/hpo.py sets for
    SHAP runs. ``executor_`` does not exist until after fit.
    """
    executor = getattr(model, "executor_", None)
    if executor is None or getattr(model, "fit_mode", None) != "fit_with_cache":
        return
    if hasattr(executor, "keep_cache_on_device"):
        executor.keep_cache_on_device = True


class TabPFNModel:
    def __init__(self, random_state=42, fit_mode=None):
        kwargs = {} if fit_mode is None else {"fit_mode": fit_mode}
        # Pinned to 2.5 rather than left on the package default: the SHAP path
        # (src/shap.py) explains TabPFN with shapiq's imputation
        # explainer, which 2.6 does not support.
        #self.model = TabPFNRegressor(fit_mode="low_memory").create_default_for_version(ModelVersion.V2_5)
        self.model = TabPFNRegressor(model_path="tabpfn-v3-regressor-v3_20260506_ood.ckpt", fit_mode="fit_with_cache",inference_config={"PASSTHROUGH_INF": True},
)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class FinetunedTabPFNModel:
    def __init__(self, random_state=42, device="cuda", epochs=30, learning_rate=1e-5, fit_mode=None):
        # No version pin and no fit_mode here, unlike TabPFNModel: this class
        # builds its inner estimator itself, hardcoding both
        # (tabpfn/finetuning/finetuned_regressor.py::_create_estimator pins
        # version=ModelVersion.V2_5 and fit_mode="batched"). Its __init__ takes
        # no **kwargs, so forwarding fit_mode raises TypeError, and routing it
        # through extra_regressor_kwargs collides with the hardcoded value.
        # Consequence: the KV cache is unavailable on this path, so SHAP over a
        # finetuned TabPFN stays slow.
        self.model = FinetunedTabPFNRegressor(
            device=device,
            epochs=epochs,
            learning_rate=float(learning_rate),
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
