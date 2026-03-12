from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline



class ElasticRegressionModel:
    def __init__(self, best_l1_ratio=0.1, best_alpha=0.06951927961775606, random_state=42):
        self.model = ElasticNet(l1_ratio=best_l1_ratio, alpha=best_alpha, max_iter=10000, random_state=random_state)
        # Create final pipeline with best parameters

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
