import numpy as np
from utils.validation import _num_samples


def sigmoid(X):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-X))


def log_logistic(X):
    """计算log(1 / (1 + e ** -X))

    Sigmoid函数的对数值
    同时参照sklearn，这里在X_i < 0时，转化为X_i + log(1 + e ** X_i)
    :param X:
    :return:
    """
    n_samples = _num_samples(X)
    out = np.zeros(n_samples)
    for i, x in enumerate(X):
        if x > 0:
            out[i] = -np.log(1 + np.exp(-x))
        else:
            out[i] = x + np.log(1 + np.exp(x))
    return out


if __name__ == '__main__':
    a = np.arange(10) - 5
    print(log_logistic(a))