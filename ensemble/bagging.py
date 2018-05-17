from base import clone
from model_selection._split import BootStrapping
import numpy as np
from utils.validation import _num_samples

class BaggingClassifier(object):
    def __init__(self, base_estimator=None, n_estimators=10):
        if not hasattr(base_estimator, "fit"):
            raise TypeError()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        bootstrap = BootStrapping()
        estimators = list()
        for i in range(self.n_estimators):
            indices, _ = bootstrap.split(X)
            estimator = clone(self.base_estimator)
            estimator.fit(X[indices], y[indices])
            estimators.append(estimator)
        self.estimators_ = estimators