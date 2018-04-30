import numpy as np
import pandas as pd
import queue
from sklearn.datasets import load_iris
from sklearn import preprocessing
from utils.validation import Check_X_y
from utils.validation import check_array
import matplotlib.pyplot as plt


def _regular(x):
    if len(x.shape) == 1:
        sum = np.sqrt(np.sum(x))
        return x / sum
    n_features = x.shape[1]
    x_square = x * x
    row_sum = np.sqrt(np.sum(x_square, axis=1))
    row_sum = np.tile(row_sum, (n_features, 1)).T
    return x / row_sum

def _sigmoid(x):
    """sigmoid函数

    sigmoid函数在几何上的意义为：对点A，有x = Aw，abs(x)越大表明离决策面越远
    此时若x > 0，则表明取1的概率越大，若x < 0，则表明取1的概率越小（取0的概率越大）
    即sigmoid函数代表了一个点取1的概率大小

    sigmoid函数弱化了边缘点对决策面带来的较大影响
    """
    res = np.zeros(x.shape)
    res1 = np.zeros(x.shape)
    idx = x > 0
    res1[idx] = 1 / (1 + np.exp(-x[idx]))
    res1[~idx] = np.exp(x[~idx]) / (np.exp(x[~idx] + 1))
    # for i, each in enumerate(x):
    #     if each > 0:
    #         res[i] = 1 / (1 + np.exp(-each))
    #     else:
    #         res[i] = np.exp(each) / (np.exp(each) + 1)
    return res1

def _intercept_dot(w, X, y):
    """计算y * np.dot(X, w)

    :return: z返回np.dot(X,w), yz返回y * z
    """
    if len(X.shape) == 1:
        z = X * w
    else:
        z = np.dot(X, w)
    yz = y * z
    return z, yz


def _logistic_loss_and_grad(w, X, y, p=None):
    """计算对率回归的损失与梯度

    与sklearn采用仿射函数及L2正则不同，这里的损失不考虑超平面线性分割及减少过拟合影响
    损失函数的形式与西瓜书不太一样，个人认为这种形式更容易理解一些（可参考吴恩达视频）

    这里需要注意的一点是西瓜书在求损失时并没有除以样本个数，牛顿法等优化算法或许的确不需要
    考虑这些，但对于梯度下降法最好还是加上，因为这影响到梯度下降时的步长取值。
    损失函数：
        loss = -sum(y * log(_sigmoid(Xw)) + (1 - y) * (log(1 - _sigmoid(Xw)))) / m
    梯度：
        grad = X.T * (_sigmoid(Xw) - y)
    :param w:
    :param X:
    :param y:
    :param alpha:
    :return:
    """
    if X.dtype == 'object':
        X.astype('float')
    z, yz = _intercept_dot(w, X, y)
    m = X.shape[0] if len(X.shape) > 1 else 1
    idz = z > 0
    # loss = -np.sum(y * np.log(_sigmoid(z)) + (1 - y) * (np.log(1 - _sigmoid(z)))) / m
    loss = 0
    if idz.any():
        loss += -np.sum(y[idz] * np.log(_sigmoid(z[idz])) + (1 - y[idz]) * (-z[idz] - np.log(np.exp(-z[idz]) + 1))) / m
        # loss += -np.sum(y[idz] * np.log(_sigmoid(z[idz])) + (1 - y[idz]) * (np.log(1 - _sigmoid(z[idz])))) / m
    if (~idz).any():
        loss += -np.sum(y[~idz] * (z[~idz] - np.log(np.exp(z[~idz]) + 1)) - (1 - y[~idz]) * np.log(1 + np.exp(z[~idz]))) / m
    if p:
        grad = X[p].T * (_sigmoid(z)[p] - y[p])
    else:
        grad = np.dot(X.T, (_sigmoid(z) - y)) / m
    return loss, grad


def _bgd(w, X, y, tol=1e-8, max_iter=400, h=0.1, return_losses=False):
    """批量梯度下降

    :param w:
    :param X:
    :param y:
    :param tol:
    :param max_iter:
    :return: coef_
    """
    losses = list()
    curr_iter = 0
    pre_w = w
    while curr_iter < max_iter:
        loss, grad = _logistic_loss_and_grad(w, X, y)
        if not losses:
            pass
        elif loss > losses[-1]:
            h /= 3
            w = pre_w - h * pre_grad
            continue
            # raise RuntimeError("Algorithms false, because loss ascent.")
        elif losses[-1] - loss < tol:
            losses.append(loss)
            break
        losses.append(loss)
        pre_w = np.array(w)
        w -= h * grad
        pre_grad = np.array(grad)
        curr_iter += 1
    if losses[-2] - losses[-1] > tol:
        print("iterator number is not enough.")
    if return_losses:
        return w, losses
    else:
        return w


def _sgd(w, X, y, tol=1e-4, max_iter=100, h=0.1, return_losses=False):
    """随机梯度下降

    :param w:
    :param X:
    :param y:
    :param tol:
    :param max_iter:
    :return:
    """
    losses = list()
    curr_iter = 0
    m = X.shape[0] if len(X.shape) > 1 else 1
    flag = False
    while curr_iter < max_iter and not flag:
        p = np.arange(m)
        np.random.shuffle(p)
        for each in p:
            loss, grad = _logistic_loss_and_grad(w, X, y, each)
            if losses and abs(losses[-1] - loss) < tol:
                flag = True
            losses.append(loss)
            w -= h * grad
            if flag:
                break
        curr_iter += 1
    if abs(losses[-2] - losses[-1]) > tol:
        print("iterator number is not enought.")
    if return_losses:
        return w, losses
    else:
        return w


def _newton_cg(w, X, y, tol=1e-4, max_iter=100, return_losses=False):
    """牛顿法


    """
    losses = list()
    n_samples = X.shape[0] if len(X.shape) > 1 else 1
    n_features = X.shape[1]
    curr_iter = 0
    while curr_iter < max_iter:
        loss, fir_grad = _logistic_loss_and_grad(w, X, y)
        if not losses:
            pass
        elif loss > losses[-1]:
            pass
            # raise RuntimeError("Algorithms false.")
        elif losses[-1] - loss < tol:
            losses.append(loss)
            break
        losses.append(loss)
        z, _ = _intercept_dot(w, X, y)
        temp = np.zeros((n_features, n_features, n_samples))
        for i in range(n_samples):
            temp[:, :, i] = np.dot(X[i, :].reshape(-1, 1), X[i, :].reshape(1, -1))
        sec_grad = np.dot(temp, _sigmoid(z) * (1 - _sigmoid(z))) / n_samples
        w -= np.dot(np.linalg.inv(sec_grad), fir_grad)
        # w = _regular(w)
        curr_iter += 1
    if losses[-2] - losses[-1] > tol:
        print("Iterator number is not enough.")
    if return_losses:
        return w, losses
    else:
        return w


class LogisticRegression(object):
    """对率回归判断器

    损失函数无正则项，solver采用批量梯度下降（BGD），随机梯度下降（SGD）及牛顿迭代（newton-cg）
    """
    def __init__(self, tol=1e-4, fit_intercept=True, intercept_scaling=1, solver='BGD', max_iter=100):
        if solver.lower() not in ['bgd', 'sgd', 'newton-cg']:
            raise ValueError("Logistic Regression supports only BGD, SGD and newton-cg, got %s" % solver)
        self.tol = tol
        self.fit_intercept=fit_intercept
        self.intercept_scaling = intercept_scaling
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y, h=0.1):
        X, y = Check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        fit_intercept = self.fit_intercept
        if fit_intercept:
            X = np.column_stack((X, self.intercept_scaling * np.ones(n_samples)))
            n_features += 1
        # X = _regular(X)
        w = np.zeros(n_features)
        # w = np.random.rand(3)
        # w = np.array([0.2, 0.2, -24])

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs at least 2 classes in the data, but the data contains"
                             " only one class: %r" % classes_[0])
        if n_classes == 2:
            n_classes = 1
            y_bin = np.ones(y.shape)
            y_bin[y == classes_[0]] = 0
            y_bin[y == classes_[1]] = 1
            if self.solver.lower() == 'bgd':
                self.coef_ = _bgd(w, X, y_bin, self.tol, self.max_iter, h)
            elif self.solver.lower() == 'sgd':
                self.coef_ = _sgd(w, X, y_bin, self.tol, self.max_iter, h)
            else:
                self.coef_ = _newton_cg(w, X, y_bin, self.tol, self.max_iter)
        else:
            self.coef_ = np.ones((n_classes, n_features))
            for i, class_ in enumerate(classes_):
                y_bin = np.ones(y.shape)
                mask = (y == class_)
                y_bin[mask] = 1
                y_bin[~mask] = 0
                if self.solver.lower() == 'bgd':
                    self.coef_[i, :] = _bgd(w, X, y_bin, self.tol, self.max_iter, h)
                    w = np.zeros(n_features)
                elif self.solver.lower() == 'sgd':
                    self.coef_[i, :] = _sgd(w, X, y_bin, self.tol, self.max_iter, h)
                    w = np.zeros(n_features)
                else:
                    self.coef_[i, :] = _newton_cg(w, X, y_bin, self.tol, self.max_iter)
                    w = np.zeros(n_features)
                    # w = np.random.rand(3)

        self.coef_ = np.array(self.coef_).reshape(n_classes, n_features)

        if fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

    def predict(self, X):
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise TypeError("This {}s instance is not fitted yet.".format(type(self).__name__))
        X = check_array(X)
        n_features = self.coef_.shape[1]
        if not n_features == X.shape[1]:
            raise ValueError("X has %d features per sample; expecting %d" % (X.shape[1], n_features))
        if not hasattr(self, 'intercept_') or self.intercept_ is None:
            predict_y = np.dot(X, self.coef_.T)
        else:
            predict_y = np.dot(X, self.coef_.T) + self.intercept_scaling * self.intercept_
        # predict_y = _sigmoid(predict_y)
        res = np.zeros(X.shape[0])
        for i, each in enumerate(predict_y):
            res[i] = self.classes_[each == each.max()]
        return res


if __name__ == '__main__':
    # watermelon_data = pd.read_csv(r"water_melon.csv", encoding='gbk')
    # X = watermelon_data.values[:, :-1].astype('float')
    # y = (watermelon_data.values[:, -1] == '是').astype('int')
    # w = np.zeros(X.shape[1])
    iris = load_iris()
    # X = iris.data[50:, :2]
    # y = iris.target[50:]
    X = iris.data[:, :2]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    y = iris.target
    # idx = (y == 1)
    # y[idx] = 0
    # y[~idx] = 1
    # X = X[:100]
    # y = y[:100]
    data = pd.read_table(r'ex2data1.txt', header=None, sep=',')
    # X = data.values[:100, :2]
    # y = data.values[:100, 2]
    classifier1 = LogisticRegression(solver='newton-cg', max_iter=100)
    classifier1.fit(X, y, h=0.1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = classifier1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.5)
    plt.axis('tight')
    colors = [[127/255, 127/255, 227/255], [163/255, 1, 213/255], [1, 127/255, 127/255]]
    for i, color in zip([0, 1, 2], colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color)
    # classifier1.fit(X, y)
    coef = classifier1.coef_
    intercept = classifier1.intercept_
    for i, c in enumerate(coef):
        y1 = (-c[0] * x_min - intercept[i]) / c[1]
        y2 = (-c[0] * x_max - intercept[i]) / c[1]
        plt.plot([x_min, x_max], [y1, y2], ls='--', color=colors[i])
        # plt.show()
        print(-c[0] / c[1], intercept[i] / c[1])
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.show()
    # x1 * a1 + x2 * a2 + a3 = 0
    # x2 = -x1 * a1 / a2 - a3 / a2

    # classifier2 = LogisticRegression(solver='SGD', max_iter=400)
    # classifier2.fit(X, y)
    # print(classifier2.coef_)
    # for i in range(100):
    #     loss, grad = _logistic_loss_and_grad(w, X, y)
    #     w -= grad
    #     print(loss)