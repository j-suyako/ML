from base import clone
from model_selection._split import BootStrapping
import numpy as np
from tree.tree import DecisionTreeClassifier


class AdaBoostClassifier(object):
    def __init__(self, base_estimator=None, n_estimators=10):
        if base_estimator:
            self.base_estimator = base_estimator
        else:
            self.base_estimator = DecisionTreeClassifier()
        self.n_estimators = n_estimators

    def fit(self, X, y):
        estimators = list()
        n_samples = X.shape[0]
        initial_weights = 1 / n_samples * np.ones(n_samples)
        for i in range(self.n_estimators):
            indices, _ = BootStrapping.split(X, sample_weights=initial_weights)
            estimator = clone(self.base_estimator)
            estimator.fit(X[indices], y[indices])
            estimators.append(estimator)
        self.estimators_ = estimators

    def predict(self, X):
        pass