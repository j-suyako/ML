import numpy as np
from utils.validation import _num_samples

def logistic(X):
    """激活函数

    公式形式为1 / (1 + exp(-X))
    :param X:
    :return:
    """
    index = (X > 0)
    X[index] = 1 / (1 + np.exp(-X[index]))
    X[~index] = 1 - 1 / (1 + np.exp(X[~index]))
    return X

def shuffle(X, y, max_iter=100):
    n_samples = _num_samples(X)
    for iter in range(max_iter):
        index = np.arange(n_samples)
        np.random.shuffle(index)
        yield X[index], y[index]