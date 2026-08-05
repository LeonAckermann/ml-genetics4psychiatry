"""Pre-flight diagnostics for whitening a phenotype covariance/intercept matrix.

Reframes "kappa needs to be low" into the question that actually matters:
is ||E||_2 / lambda_min small, i.e. is the bottom of the estimated spectrum
distinguishable from estimation noise. See whitening/README.md for the full
checklist and whitening/run.py for the entry point.
"""

