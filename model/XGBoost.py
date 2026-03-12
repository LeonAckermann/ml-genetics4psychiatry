import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


class XGBoostTreeModel:
    def __init__(self, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8):
        self.model = xgb.XGBRegressor(
            random_state=random_state,
            verbosity=0,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)