"""Bayesian ridge regression: Gaussian prior over the weights.

Model:
    y | X, w, alpha  ~  N(X w, alpha^-1 I)          # Gaussian likelihood
    w | lambda       ~  N(0, lambda^-1 I)           # isotropic Gaussian prior

The prior is the Bayesian counterpart of the L2 penalty in ``Ridge``: the MAP
estimate of ``w`` equals the ridge solution with alpha_ridge = lambda / alpha.
The difference is that the precisions are not fixed by cross-validation --
both get Gamma hyperpriors and are estimated on the training fold by evidence
maximisation (type-II maximum likelihood), so this model has no penalty
strength to tune.

Prediction is the posterior predictive mean

    y_mean = X w_mean

i.e. the weights are integrated out rather than plugged in at a point
estimate. The posterior predictive standard deviation

    y_std = sqrt(1/alpha + diag(X Sigma X^T))

is available separately via :meth:`predict_std` / :meth:`predict_with_std`;
the pipeline scores ``predict`` alone, which returns the mean.
"""
from sklearn.linear_model import BayesianRidge


class BayesianRidgeRegressionModel:
    """Wrapper around ``sklearn.linear_model.BayesianRidge``.

    Parameters
    ----------
    alpha_1, alpha_2 : Gamma(shape, rate) hyperprior on the noise precision
        alpha. The sklearn defaults (1e-6, 1e-6) are near-flat, which is what
        we want -- the noise level should come from the data.
    lambda_1, lambda_2 : Gamma(shape, rate) hyperprior on the weight-prior
        precision lambda. Also near-flat by default. Raising ``lambda_1``
        pushes lambda up, i.e. tightens the Gaussian prior around zero and
        shrinks the coefficients harder.
    max_iter, tol : evidence-maximisation convergence controls.
    fit_intercept : the intercept is not penalised by the prior (sklearn
        centres the data internally), so leave it on unless the target is
        already centred.

    Features must be standardised before fitting -- the prior is isotropic, so
    it only shrinks coefficients comparably when the columns share a scale.
    The pipeline handles this: ``bayesian_ridge_regression`` is in
    ``src.hpo.NEEDS_SCALING``, which applies a per-fold ``StandardScaler``.
    """

    def __init__(self, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6,
                 max_iter=300, tol=1e-3, fit_intercept=True):
        self.model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """Posterior predictive mean ``y_mean`` -- the pipeline's prediction."""
        return self.model.predict(X_test)

    def predict_with_std(self, X_test):
        """``(y_mean, y_std)``: the predictive mean and its standard deviation.

        ``y_std`` combines the estimated noise level with the uncertainty in
        ``w``, so it widens where the test point is far from the training
        design. Not used for scoring; kept for calibration / uncertainty work.
        """
        return self.model.predict(X_test, return_std=True)

    def predict_std(self, X_test):
        """Posterior predictive standard deviation only."""
        return self.model.predict(X_test, return_std=True)[1]

    @property
    def coef_(self):
        """Posterior mean of the weights."""
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    @property
    def alpha_(self):
        """Estimated noise precision (1/alpha_ is the noise variance)."""
        return self.model.alpha_

    @property
    def lambda_(self):
        """Estimated prior precision over the weights.

        ``lambda_ / alpha_`` is the equivalent ``Ridge(alpha=...)`` penalty
        the evidence selected, which makes it directly comparable to the
        alpha tuned by CV for ``ridge_regression``.
        """
        return self.model.lambda_

    @property
    def sigma_(self):
        """Posterior covariance of the weights."""
        return self.model.sigma_
