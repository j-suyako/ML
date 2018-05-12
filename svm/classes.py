import numpy as np
from utils.validation import _num_samples

from ._base import select, linear


class SVC(object):

    def __init__(self, C, kernel='linear', tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.tol = tol

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
        self.alphas = np.zeros(n_samples)
        self.b = 0
        C = self.C
        self.w = np.dot(X.T, self.alphas * y)
        u = np.dot(X, self.w) - self.b
        self.error = u - y
        for iter_num in range(100):
            if iter_num:
                index = np.where(0 < self.alphas < C)  # 提取所有可能的支持向量，之后的alpha更新都在这些向量中进行
            else:
                index = np.arange(n_samples)
            alphas = self.alphas[index]  # python浅拷贝机制，在alphas中任意元素的改变都会导致self.alphas对应元素的改变
            X_curr = X[index]
            y_curr = y[index]
            u_curr = np.dot(X_curr, self.w) - self.b
            error_curr = u_curr - y_curr
            self.error[index] = error_curr
            r_curr = error_curr * y_curr
            for i, alpha1 in enumerate(alphas):
                # i的启发式选择,显然，在这里，当alpha1不等于C或者0时，意味着X[i]较可能为支持向量，则我们改变
                # alpha[i]所带来的受益会更大（alpha[i]与w息息相关，对非支持向量更新alpha后可能依然为C或者0，
                # 这时w不会改变）。同时，因为支持向量满足uy=1，所以如果已经满足uy=1则不进行更新（给了一个tol的
                # 允许误差范围）
                # if (r_curr[i] < - self.tol and alpha1 < C) or (r_curr[i] > self.tol and alpha1 > 0):
                if abs(r_curr[i]) > self.tol:
                    fai = np.dot(self.w.T, self.w) / 2 - np.sum(alphas)
                    j = select(i, X_curr, error_curr)
                    alpha2 = alphas[j]
                    K = (linear(X_curr[i], X_curr[i]), linear(X_curr[i], X_curr[j]), linear(X_curr[j], X_curr[j]))
                    alphas[i], alphas[j] = self._update_alpha(i, j, y_curr, K, error_curr, alphas)
                    delta_alpha = alphas[i] - alpha1, alphas[j] - alpha2
                    self.b = self._update_b(i, j, y_curr, K, delta_alpha, error_curr, alphas)
                    self.w = self._update_w(i, j, X_curr, y_curr, delta_alpha)
                    # u = np.dot(X, self.w) - self.b
                    # self.error = u - y

    def _update_alpha(self, i, j, y, K, error, alphas):
        """update the alpha about i and j

        :return:
        """
        C = self.C
        y1, y2 = y[i], y[j]
        error1, error2 = error[i], error[j]
        alpha1, alpha2 = alphas[i], alphas[j]
        if y1 == y2:
            low = max(0, alpha2 + alpha1 - C)
            high = min(C, alpha2 + alpha1)
        else:
            low = max(0, alpha2 - alpha1)
            high = min(C, C + alpha2 - alpha1)
        if low == high:
            return alpha1, alpha2
        s = y1 * y2
        k11, k12, k22 = K[0], K[1], K[2]
        enta = k11 + k22 - 2 * k12  # enta为cost关于alpha2的二阶导数
        if enta > 0:  # enta大于0意味着cost存在最小解
            alpha2_new = self._clip_alpha(alpha2 + y2 * (error1 - error2) / enta, low, high)
        else:
            f1 = y1 * (error1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (error2 + self.b) - s * alpha1 * k12 - alpha2 * k22
            l1 = alpha1 + s * (alpha2 - low)
            h1 = alpha1 + s * (alpha2 - high)
            fai__l = l1 * f1 + low * f2 + l1 ** 2 * k11 / 2 + low ** 2 * k22 / 2 + s * low * l1 * k12
            fai__h = h1 * f1 + high * f2 + h1 ** 2 * k11 / 2 + high ** 2 * k22 / 2 + s * high * h1 * k12
            if fai__l < fai__h:
                alpha2_new = low
            elif fai__l > fai__h:
                alpha2_new = high
            else:
                alpha2_new = alpha2
        if alpha2 == alpha2_new:
            return alpha1, alpha2
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)
        return alpha1_new, alpha2_new

    def _update_b(self, i, j, y, K, delta_alpha, error, alphas):
        C = self.C
        k11, k12, k22 = K[0], K[1], K[2]
        delta_alpha1, delta_alpha2 = delta_alpha[0], delta_alpha[1]
        b1 = error[i] + y[i] * delta_alpha1 * k11 + y[j] * delta_alpha2 * k12 + self.b
        b2 = error[j] + y[i] * delta_alpha1 * k12 + y[j] * delta_alpha2 * k22 + self.b
        if 0 < alphas[i] < C:
            # this condition means X[i] is the support vector, thus w.T * X[i] + b = y[i]
            return b1
        elif 0 < alphas[j] < C:
            # this condition means X[j] is the support vector, thus w.T * X[j] + b = y[j]
            return b2
        else:
            # this means X[i] and X[j] neither to be the support vector, then the hyperplane between
            # with the hyperplanes corresponding to X[i] & X[j] are consistent with the KKT conditions.
            return (b1 + b2) / 2

    def _update_w(self, i, j, X, y, delta_alpha):
        return self.w + y[i] * delta_alpha[0] * X[i] + y[j] * delta_alpha[1] * X[j]
