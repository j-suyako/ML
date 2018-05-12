import numpy as np
from utils.validation import _num_samples


def select(i, X, error):
    """启发式选择第二个代更新alpha的index

    选择方式依据如下：
    由于alpha2_new = alpha2 + y2 * (error1 - error2) / enta，所以选择abs(error1 - error2)更大
    的更新alpha的幅度也大->损失函数下降越快
    :param i:
    :param X:
    :param error:
    :return:
    """
    if not isinstance(i, int):
        raise TypeError()
    # n_samples = _num_samples(X)
    if error[i] > 0:
        j = np.where(error == min(error))
    else:
        j = np.where(error == max(error))
    # j = i
    # while j == i:
    #     j = np.random.randint(0, n_samples)
    return j


def linear(x1, x2):
    return np.dot(x1, x2)