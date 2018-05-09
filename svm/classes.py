import numpy as np
from utils.validation import _num_samples

class SVC(object):

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        n_samples = _num_samples(X)
        alpha = np.zeros(n_samples)
        w = np.dot(X.T, alpha * y)
        cost = np.dot(w.T, w) / 2 - np.sum(alpha)
