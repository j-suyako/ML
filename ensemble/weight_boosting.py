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

    def additive(self, X):
        H = np.zeros(X.shape[0])
        for estimator, alpha in zip(self.estimators_, self.alphas_):
            H += alpha * estimator.predict(X)
        return H

    def fit(self, X, y):
        self.estimators_ = list()
        n_samples = X.shape[0]
        self.alphas_ = list()
        weights = 1 / n_samples * np.ones(n_samples)
        cost = list()
        for i in range(self.n_estimators):
            indices, _ = BootStrapping.split(X, sample_weights=weights)
            estimator = clone(self.base_estimator)
            estimator.fit(X[indices], y[indices])
            predict_y = estimator.predict(X)
            expect = np.sum(weights[predict_y - y != 0])
            if expect > 0.5:
                i -= 1
                continue
            alpha = 0.5 * np.log((1 - expect) / expect)
            self.alphas_.append(alpha)
            self.estimators_.append(estimator)
            H = self.additive(X)
            weights = np.exp(-y * H) / np.sum(np.exp(-y * H))
            cost.append(np.sum(np.exp(-y * H)) / n_samples)

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise ValueError()
        H = self.additive(X)
        H[H > 0] = 1
        H[H < 0] = -1
        return H
