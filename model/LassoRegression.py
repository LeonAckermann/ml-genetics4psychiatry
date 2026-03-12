from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline



class LassoRegressionModel:
    def __init__(self, best_alpha=0.02123011124333675, random_state=42):
        self.model = Lasso(alpha=best_alpha, max_iter=10000, random_state=random_state)
        # Create final pipeline with best alpha

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
