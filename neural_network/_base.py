import numpy as np

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