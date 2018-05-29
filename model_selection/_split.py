import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from utils.validation import _num_samples

__all__ = ['KFold',
           'LeaveOneOut',
           'BootStrapping',
           'train_test_split']


class KFold(object):
    """K折交叉验证

    将数据集划分为k个大小相似的互斥子集，每次用k-1个子集作为训练集，余下的子集作为测试集；最终得到
    k组训练/测试集。

    Parameters
    ----------
    n_splits : int, default=3
        折数，最小为2
    shuffle : boolean, optional
        是否在划分之前对数组进行重排
    """

    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, X):
        """每次生成训练/测试集的索引

        :param X: 样本
        :return: 训练/测试集索引
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be none.")
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        n_splits = self.n_splits
        fold_sizes = n_samples // n_splits * np.ones(n_splits, dtype='int')
        fold_sizes[:n_samples % n_splits] += 1
        start = 0
        for i in range(n_splits):
            end = start + fold_sizes[i]
            yield np.concatenate((indices[:start], indices[end:])), indices[start:end]
            start = end


class LeaveOneOut(object):
    """留一法

    每次取出一个样本作为测试集，其余作为训练集
    """
    def split(self, X):
        if X is None:
            raise ValueError("The 'X' parameter should not be none.")
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        for i in range(X.shape[0]):
            yield np.concatenate((indices[:i], indices[i + 1:])), indices[i]


class BootStrapping(object):
    """自助法

    有放回取样，由概率论相关知识可知约有1/e的样本永远不会采集到，这部分样本作为
    测试集，其余被采集的样本作为训练集（显然此时训练集中存在重复样本）

    自助法适用于数据集较小，难以有效划分训练/测试集时比较有用，但自助法改变了初始
    数据集的分布，因此在数据量足够时应采用留出法和交叉验证法
    """
    @staticmethod
    def split(X, sample_weights=None, iter_num=None):
        if X is None:
            raise ValueError("The 'X' parameter should not be none.")
        n_samples = _num_samples(X)
        if iter_num is None:
            iter_num = n_samples
        if sample_weights is not None:
            weights = sample_weights / sum(sample_weights)
            indices = list(map(BootStrapping.func, BootStrapping.loop(weights, iter_num)))
        else:
            indices = list(map(np.random.randint, BootStrapping.loop(n_samples, iter_num)))
        return indices, list(set(np.arange(n_samples)) - set(indices))

    @staticmethod
    def func(sample_weights):
        threshold = np.random.random() * sum(sample_weights)
        curr = 0
        index = 0
        while curr < threshold:
            curr += sample_weights[index]
            index += 1
        return index - 1

    @staticmethod
    def loop(data, iter_num):
        for _ in range(iter_num):
            yield data


def train_test_split(X, y, test_size=0.3, shuffle=False):
    """留出法

    只是在使用形式上与sklearn的函数一致，没有原函数那么复杂。
    :param X:
    :param y:
    :param test_size:
    :param shuffle:
    :return:
    """
    if not _num_samples(X) == _num_samples(y):
        raise ValueError("The 'X' and 'y' should be equivalent in first dimension")
    indices = np.arange(_num_samples(y))
    if shuffle:
        np.random.shuffle(indices)
    train_num = np.floor((1 - test_size) * len(y)).astype(int)
    X_train, X_test = X[indices[:train_num]], X[indices[train_num:]]
    y_train, y_test = y[indices[:train_num]], y[indices[train_num:]]
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # iris_data = pd.read_csv('../data/iris.csv',
    #                         names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    # X = np.column_stack((iris_data.values[:100, :4], np.ones(100))).astype(float)
    iris_data = load_iris()
    X = iris_data.data
    # y = pd.Series(iris_data['class'] == 'Iris-setosa', dtype=int)[:100]
    kf = KFold(n_splits=10, shuffle=True)
    for i, (train, test) in enumerate(kf.split(X)):
        if i == 9:
            print(X[train], X[test])