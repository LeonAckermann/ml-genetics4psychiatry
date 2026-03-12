from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline



class RidgeRegressionModel:
    def __init__(self, best_alpha=1000.0, random_state=42):
        self.model = Pipeline([
            ("ridge", Ridge(alpha=best_alpha, random_state=random_state))
        ])
        # Create final pipeline with best alpha

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
