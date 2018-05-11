import numpy as np
from utils.validation import _num_samples

from ._base import select, linear

class SVC(object):

    def __init__(self, C):
        self.C = C

    @staticmethod
    def _clip_alpha(alpha_new, low, high):
        if alpha_new <= low:
            return low
        elif alpha_new >= high:
            return high
        return alpha_new

    def fit(self, X, y):
        """暂时只写二分类，y为1或者-1

        view the ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf(original paper)
        or the ref: http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html(in chinese)
        :param X:
        :param y:
        :return:
        """
        n_samples = _num_samples(X)
        alphas = np.zeros(n_samples)
        b = 0
        C = self.C
        w = np.dot(X.T, alphas * y)
        for iter_num in range(100):
            for i, alpha1 in enumerate(alphas):
                cost = np.dot(w.T, w) / 2 - np.sum(alphas)
                u1 = np.dot(X[i, :], w) - b
                E1 = u1 - y[i]
                j = select(i, X)  # TODO(suyako): 之后采用启发式，当前为随机选择
                alpha2 = alphas[j]
                u2 = np.dot(X[j, :], w) - b
                E2 = u2 - y[j]
                if y[i] == y[j]:
                    L = max(0, alpha2 + alpha1 - C)
                    H = min(C, alpha2 + alpha1)
                else:
                    L = max(0, alpha2 - alpha1)
                    H = min(C, C + alpha2 - alpha1)
                K11 = linear(X[i, :], X[i, :])
                K22 = linear(X[j, :], X[j, :])
                K12 = linear(X[i, :], X[j, :])
                enta = K11 + K22 - 2 * K12  # 二阶导数
                alphas[j] = self._clip_alpha(alpha2 + y[j] * (E1 - E2) / enta, L, H)
                alphas[i] += y[i] * y[j] * (alpha2 - alphas[j])
                b1 = E1 + y[i] * (alphas[i] - alpha1) * K11 + y[j] * (alphas[j] - alpha2) * K12 + b
                b2 = E2 + y[i] * (alphas[i] - alpha1) * K12 + y[j] * (alphas[j] - alpha2) * K22 + b
                if 0 < alphas[i] and C > alphas[i]:
                    # this condition means X[i] is the support vector, thus w.T * X[i] + b = y[i]
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    # this condition means X[j] is the support vector, thus w.T * X[j] + b = y[j]
                    b = b2
                else:
                    # this means X[i] and X[j] neither to be the support vector, then the hyperplane between
                    # with the hyperplanes corresponding to X[i] & X[j] are consistent with the KKT conditions.
                    b = (b1 + b2) / 2
                w += y[i] * (alphas[i] - alpha1) * X[i, :] + y[j] * (alphas[j] - alpha2) * X[j, :]
        w = np.dot(X.T, alphas * y)
        return w
