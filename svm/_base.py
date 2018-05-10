import numpy as np
from utils.validation import _num_samples

def select(i, X):
    if not isinstance(i, int):
        raise TypeError()
    n_samples = _num_samples(X)
    index = i
    while index == i:
        index = np.random.randint(0, n_samples)
    return index

def linear(x1, x2):
    return np.dot(x1, x2)