from base import clone
from model_selection._split import BootStrapping
import numpy as np
from tree.tree import DecisionTreeClassifier
from utils.validation import _num_samples


class BaggingClassifier(object):

    def __init__(self, base_estimator=None, n_estimators=10):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
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

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise TypeError()
        res = np.zeros((_num_samples(X), 1))
        y = np.zeros((_num_samples(X), self.n_estimators))
        for i, estimator in enumerate(self.estimators_):
            if not hasattr(estimator, "predict"):
                raise TypeError()
            y[:, i] = estimator.predict(X)
        for i, line in enumerate(y):
            res[i] = np.argmax(np.bincount(line))
        return res