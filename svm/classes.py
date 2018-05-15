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

        # i的启发式选择,显然，在这里，当alpha1不等于C或者0时，意味着X[i]较可能为支持向量，则我们改变
        # alpha[i]所带来的受益会更大（alpha[i]与w息息相关，对非支持向量更新alpha后可能依然为C或者0，
        # 这时w不会改变）。同时，因为支持向量满足uy=1，所以如果已经满足uy=1则不进行更新（给了一个tol的
        # 允许误差范围）

        view the ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf(original paper)
        or the ref: http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html(in chinese)
        :param X:
        :param y:
        :return:
        """
        n_samples = _num_samples(X)
        self.alphas = np.zeros(n_samples)
        self.w = np.dot(X.T, self.alphas * y)
        self.b = 0
        self.errors = np.dot(X, self.w) - self.b - y
        self.fai = list()
        C = self.C
        num_changed = 0
        examine_all = True
        iter_num = 0
        while iter_num < 100 and (num_changed > 0 or examine_all):
            num_changed = 0
            if examine_all:
                num_changed = sum(self.examineExample(i, X, y) for i in range(n_samples))
            else:
                num_changed = sum(self.examineExample(i, X, y) for i in range(n_samples)
                                  if self.alphas[i] != 0 and self.alphas[i] != C)
            if examine_all:
                examine_all = False
            elif not num_changed:
                examine_all = True
            iter_num += 1

        # u = np.dot(X, self.w) - self.b
        # fai = []
        # self.r = (u - y) * y

        pass

    def examineExample(self, i, X, y):
        """

        第一个if语句中为KKT条件判定，首先要明白alpha必定在0与C之间（初始化时alpha为0），然后分析如下：
        1. 当alpha为0时，可知其对应的向量在支持向量外，KKT条件为uy>1，所以应满足uy-1>0
        2. 当alpha为C时，可知其对应的向量在支持向量内，KKT条件为uy<1，所以应满足uy-1<0
        3. 当alpha为0或C时，可知其为支持向量，KKT条件为uy=1，所以应满足uy-1=0
        上述三个条件在实际计算中给了一个tol的容忍度

        :param i:
        :param X:
        :param y:
        :return:
        """
        y_i = y[i]
        alphas = self.alphas
        alpha1 = alphas[i]
        # u_i = np.dot(X[i].T, self.w) - self.b
        error_i = self.errors[i]
        r_i = error_i * y_i
        not_bound_alpha_index = self._get_none_bound_alpha_index()
        if (r_i < -self.tol and alpha1 < self.C) or (r_i > self.tol and alpha1 > 0):
            if len(not_bound_alpha_index) > 1:
                j = select(i, self.errors, not_bound_alpha_index)
                if self._updated(i, j, X, y):
                    return 1
            for j in not_bound_alpha_index:
                if self._updated(i, j, X, y):
                    return 1
            all_index = np.arange(_num_samples(alphas))
            np.random.shuffle(all_index)
            for j in all_index:
                if self._updated(i, j, X, y):
                    return 1
        return 0

    def _updated(self, i, j, X, y):
        """update the alpha about i and j

        :return:
        """
        if i == j:
            return False
        C = self.C
        y1, y2 = y[i], y[j]
        errors = self.errors
        error1, error2 = errors[i], errors[j]
        alphas = self.alphas
        alpha1, alpha2 = alphas[i], alphas[j]
        if y1 == y2:
            low = max(0, alpha2 + alpha1 - C)
            high = min(C, alpha2 + alpha1)
        else:
            low = max(0, alpha2 - alpha1)
            high = min(C, C + alpha2 - alpha1)
        if low == high:
            return False
        s = y1 * y2
        K = (linear(X[i], X[i]), linear(X[i], X[j]), linear(X[j], X[j]))
        k11, k12, k22 = K[0], K[1], K[2]
        enta = k11 + k22 - 2 * k12  # enta为损失函数关于alpha2的二阶导数
        if enta > 0:  # enta大于0意味着损失函数存在最小解
            alpha2_new = self._clip_alpha(alpha2 + y2 * (error1 - error2) / enta, low, high)
        else:  # enta小于0以为着损失函数的最小解在边界上
            f1 = y1 * (error1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (error2 + self.b) - s * alpha1 * k12 - alpha2 * k22
            l1 = alpha1 + s * (alpha2 - low)
            h1 = alpha1 + s * (alpha2 - high)
            fai__l = l1 * f1 + low * f2 + l1 ** 2 * k11 / 2 + low ** 2 * k22 / 2 + s * low * l1 * k12
            fai__h = h1 * f1 + high * f2 + h1 ** 2 * k11 / 2 + high ** 2 * k22 / 2 + s * high * h1 * k12
            if fai__l < fai__h - 0.001:
                alpha2_new = low
            elif fai__l > fai__h + 0.001:
                alpha2_new = high
            else:
                alpha2_new = alpha2
        if abs(alpha2 - alpha2_new) < 0.001 * (alpha2 + alpha2_new + 0.001):  # 这一步不太明白具体是什么操作，论文给的伪代码是这么写的
            return False
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)
        alphas[i], alphas[j] = alpha1_new, alpha2_new
        delta_alpha1, delta_alpha2 = alphas[i] - alpha1, alphas[j] - alpha2
        b1 = errors[i] + y[i] * delta_alpha1 * k11 + y[j] * delta_alpha2 * k12 + self.b
        b2 = errors[j] + y[i] * delta_alpha1 * k12 + y[j] * delta_alpha2 * k22 + self.b
        if 0 < alphas[i] < C:
            # this condition means X[i] is the support vector, thus w.T * X[i] + b = y[i]
            self.b = b1
        elif 0 < alphas[j] < C:
            # this condition means X[j] is the support vector, thus w.T * X[j] + b = y[j]
            self.b = b2
        else:
            # this means X[i] and X[j] neither to be the support vector, then the hyperplane between
            # with the hyperplanes corresponding to X[i] & X[j] are consistent with the KKT conditions.
            self.b = (b1 + b2) / 2
        self.w += y[i] * delta_alpha1 * X[i] + y[j] * delta_alpha2 * X[j]
        self.errors = np.dot(X, self.w) - self.b - y
        self.fai.append(np.dot(self.w.T, self.w) / 2 - np.sum(alphas))
        return True

    # def _update_b(self, i, j, y, K, delta_alpha, errors, alphas):
    #     C = self.C
    #     k11, k12, k22 = K[0], K[1], K[2]
    #     delta_alpha1, delta_alpha2 = delta_alpha[0], delta_alpha[1]
    #     b1 = errors[i] + y[i] * delta_alpha1 * k11 + y[j] * delta_alpha2 * k12 + self.b
    #     b2 = errors[j] + y[i] * delta_alpha1 * k12 + y[j] * delta_alpha2 * k22 + self.b
    #     if 0 < alphas[i] < C:
    #         # this condition means X[i] is the support vector, thus w.T * X[i] + b = y[i]
    #         self.b = b1
    #     elif 0 < alphas[j] < C:
    #         # this condition means X[j] is the support vector, thus w.T * X[j] + b = y[j]
    #         self.b = b2
    #     else:
    #         # this means X[i] and X[j] neither to be the support vector, then the hyperplane between
    #         # with the hyperplanes corresponding to X[i] & X[j] are consistent with the KKT conditions.
    #         self.b = (b1 + b2) / 2
    #
    # def _update_w(self, i, j, X, y, delta_alpha):
    #     self.w += y[i] * delta_alpha[0] * X[i] + y[j] * delta_alpha[1] * X[j]

    def _get_none_bound_alpha_index(self):
         flags = (self.alphas > 0) & (self.alphas < self.C)
         index = np.arange(_num_samples(self.alphas))
         index = index[flags]
         np.random.shuffle(index)
         return index