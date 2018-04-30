from abc import ABCMeta, abstractmethod
import copy
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
import sys
from utils.validation import check_array, Check_X_y, _num_samples


def _median(X):
    """生成X相邻元素中位数的数组

    一定要记得对X进行排序
    :param X: 类array数组
    :return:
    """
    n_samples = _num_samples(X)
    if n_samples == 1:
        yield X[0]
    for i in range(n_samples - 1):
        yield (X[i] + X[i + 1]) / 2
        # yield float('%.2f' % ((X[i] + X[i + 1]) / 2))

class Node(object):
    __metaclass__ = ABCMeta

    def __init__(self, X, y, attributes=None):
        X, y = Check_X_y(X, y)
        self.X = X  # 每个节点对应的训练集
        self.y = y  # 训练集对应的样本分类
        n_samples, n_features = X.shape
        classifiers, count = np.unique(y, return_counts=True)
        self.freq_max_y = classifiers[0] if len(classifiers) > 0 else classifiers # 该节点中样本最多的类，主要在predict中使用
        self.n_samples = n_samples  # 训练集数量
        self.child = {}  # 训练集的子节点，由split动态生成
        if not attributes:
            attributes = list(range(n_features))
        self.attributes = attributes  # attributes代表属性集，如果缺省的话给一个0 - N-1的列表，其中N是X的属性集数量
        features = list(range(n_features))
        self.features = features
        self.split_features = None
        # self.split_features = self.attributes[0]  # 训练集的最优划分属性
        self.classifier = classifiers[0] if len(classifiers) == 1 else None  # 该节点所对应的类别，如果y中有不止一个分类则返回None
        # self.feature_chosen = np.zeros(n_features, dtype=bool)
        self.feature_chosen = [False] * n_features  # 用来保存已经用来分割过的属性集

    @abstractmethod
    def fit(self):
        pass


class NodeByID3(Node):
    """基于ID3决策树算法的模型

    最优属性划分采用的是信息熵增益的方法（注意与C4.5信息熵增益率的区别）
    只能用于非连续属性的划分（实际上二分法对这个应该没有限制，我也不懂为什么都说ID3不能处理连续值）
    """

    def __init__(self, X, y, attributes=None):
        super(NodeByID3, self).__init__(X, y, attributes=attributes)

    def ent(self):
        """该样本集合的信息熵

        信息熵定义为 Ent(D) = -sum(p_k * log2(p_k))，其中p_k为第k类样本所占的比例
        :return:
        """
        classes_and_counts = np.unique(self.y, return_counts=True)
        entropy_of_sample = 0
        for each in classes_and_counts[1]:
            p_each = each / len(self.y)
            entropy_of_sample -= p_each * np.log2(p_each)
        return entropy_of_sample

    def _feature_choose(self, features_k):
        """选择X的第k个属性进行分割，生成Dv

        假设根据第k个属性，原样本可以被划分为m个集合，则每次for循环生成
        其中一个叶节点集合
        :param features_k:
        :return:
        """
        if features_k < 0 or features_k >= len(self.features):
            raise ValueError("No such features.")
        for attr in np.unique(self.X[:, features_k]):
            index = (self.X[:, features_k] == attr)
            X = self.X[index]  # 找到每个属性值包含的训练集
            y = self.y[index]  # 该属性值包含的训练集对应的类别
            yield attr, NodeByID3(X, y, self.features)

    def gain(self, features_k):
        """计算用第k个属性对样本进行划分得到的信息增益

        :param features_k:
        :return:
        """
        res = self.ent()
        for _, each in self._feature_choose(features_k):
            res -= each.n_samples / self.n_samples * each.ent()
        return res

    def fit(self):
        """fit操作

        :return:
        """
        # 递归返回条件，如果所有样本属于同一个类别则不做处理，相当于标记为叶节点
        if self.classifier is None:
            Gain = list()
            Gain_index = list()
            for k, _ in enumerate(self.features):
                # 如果该属性已经在之前被选择过作为划分依据，则跳过
                if self.feature_chosen[k]:
                    continue
                # if k in self.feature_chosen:
                #     continue
                Gain.append(self.gain(k))  # 计算每个属性划分所带来的信息增益并存入Gain
                Gain_index.append(k)  # 同时记录被用来试验的属性的index
            best_features = Gain_index[Gain.index(max(Gain))]  # 选择最优属性
            self.set_split_features(best_features)  # 用该最优属性进行划分
            # 在已选择属性向量中增加该最优属性（Attention！每个分支节点对应的已选择属性都是不一样的）
            # self.feature_chosen.append(best_features)
            for each_child in self.child:
                # self.child[each_child].feature_chosen = copy.copy(self.feature_chosen)
                # self.child[each_child].feature_chosen[best_features] = True
                self.child[each_child].fit()  # 递归
        # else:
        #     self.split_features = None

    def set_split_features(self, features_k):
        self.split_features = self.attributes[features_k]
        for attr, each in self._feature_choose(features_k):
            self.child[attr] = each  # child中保存由该最优属性划分得到的每个属性值对应的叶节点
            # self.child[attr].feature_chosen[self.feature_chosen] = True
            self.child[attr].feature_chosen = copy.copy(self.feature_chosen)
            self.child[attr].feature_chosen[features_k] = True

    def __repr__(self):
        info = ['splitter = {}', 'Ent = {}', 'n_samples = {}', 'class = {}']
        toshow = [self.split_features, self.ent(), self.n_samples, self.classifier]
        for i in range(4):
            info[i] = info[i].format(toshow[i])
        return '[' + '\n'.join(info) + ']'

    def __str__(self):
        return self.__repr__()


class NodeByC4_dot_5(Node):
    """基于C4.5决策树算法的模型

    最优属性划分采用信息熵增益率的方法（注意与ID3信息熵增益的区别）
    可以处理连续变量
    """
    def __init__(self, X, y, attributes=None):
        super(NodeByC4_dot_5, self).__init__(X, y, attributes=attributes)
        for i, each in enumerate(self.features):
            values = np.unique(X[:, i])
            each = list()
            for median_value in _median(values):
                each.append(median_value)
            self.features[i] = each
            self.feature_chosen[i] = [False] * len(each)


    def ent(self):
        """该样本集合的信息熵

        信息熵定义为 Ent(D) = -sum(p_k * log2(p_k))，其中p_k为第k类样本所占的比例
        :return:
        """
        classes_and_counts = np.unique(self.y, return_counts=True)
        entropy_of_sample = 0
        for each in classes_and_counts[1]:
            p_each = each / len(self.y)
            entropy_of_sample -= p_each * np.log2(p_each)
        return entropy_of_sample

    def _feature_choose(self, features_k, value_v):
        """选择X的第k个属性进行分割，生成Dk

        假设根据第k个属性，原样本可以被划分为m个集合，则每次for循环生成
        其中一个叶节点集合
        :param features_k:
        :return:
        """
        if features_k < 0 or features_k >= len(self.features):
            raise ValueError("No such features.")
        if value_v < 0 or value_v >= len(self.features[features_k]):
            raise ValueError("No such value in that features.")
        for i in range(2):
            index = (self.X[:, features_k] <= self.features[features_k][value_v])
            if not i:
                X = self.X[index]
                y = self.y[index]
                attr = "{} <= {}".format(self.attributes[features_k], self.features[features_k][value_v])
            else:
                X = self.X[~index]
                y = self.y[~index]
                attr = "{} > {}".format(self.attributes[features_k], self.features[features_k][value_v])
            yield attr, NodeByC4_dot_5(X, y, self.attributes)
        # for each_value in np.unique(self.X[:, features_k]):
        #     index = (self.X[:, features_k] == each_value)
        #     X = self.X[index]  # 找到每个属性值包含的训练集
        #     y = self.y[index]  # 该属性值包含的训练集对应的类别
        #     yield each_value, NodeByC4_dot_5(X, y, self.attributes)

    def gain(self, features_k, value_v):
        """计算用第k个属性对样本进行划分得到的信息增益

        :param features_k:
        :return:
        """
        res = self.ent()
        for _, each in self._feature_choose(features_k, value_v):
            res -= each.n_samples / self.n_samples * each.ent()
        return res

    def _IV(self, features_k):
        """属性k的固有值（intrinsic value）

        公式为IV(fea_k) = -sum(freq(Dv) * log2(freq(Dv)))，其中Dv为属性k的第v个取值，freq(Dv)代表该取值的频率
        一般来说属性值越多，IV会越大，所以基于增益率的C4.5改进了ID3偏向于选择属性值较多的属性的缺陷
        :param features_k:
        :return:
        """
        n_samples, _ = self.X.shape
        _, value_samples = np.unique(self.X[:, features_k], return_counts=True)
        value_samples_freq = value_samples / n_samples
        return -np.sum(value_samples_freq * np.log2(value_samples_freq))

    def gain_ratio(self, features_k, value_v):
        gain = self.gain(features_k, value_v)
        IV = self._IV(features_k)
        if not gain or not IV:
            return 0
        return self.gain(features_k, value_v) / self._IV(features_k)

    def fit(self):
        if self.classifier is None:
            Gain_ratio = list()
            Gain_ratio_index = list()
            for i, feature in enumerate(self.features):
                for j, value in enumerate(feature):
                    if self.feature_chosen[i][j]:
                        continue
                    Gain_ratio.append(self.gain_ratio(i, j))
                    Gain_ratio_index.append((i, j))
            features_k, value_v = Gain_ratio_index[Gain_ratio.index(max(Gain_ratio))]
            self.set_split_features(features_k, value_v)
            for each_child in self.child:
                self.child[each_child].fit()

    def set_split_features(self, features_k, value_v):
        self.split_features = (features_k, value_v)
        # self.split_features = self.features[features_k][value_v]
        for i, (attr, each) in enumerate(self._feature_choose(features_k, value_v)):
            self.child[attr] = each
            self.child[attr].feature_chosen = copy.deepcopy(self.feature_chosen)
            while value_v < len(self.feature_chosen[features_k]) and value_v >= 0:
                self.child[attr].feature_chosen[features_k][value_v] = True
                if not i:
                    value_v += 1
                else:
                    value_v -= 1


class DecisionTreeClassifier(object):

    def __init__(self, criterion='ID3', attributes=None):
        self.criterion = criterion
        self.attributes = attributes

    def fit(self, X, y):
        if self.criterion == 'ID3':
            self.tree_ = NodeByID3(X, y, attributes=self.attributes)
        elif self.criterion == 'C4.5':
            self.tree_ = NodeByC4_dot_5(X, y, attributes=self.attributes)
        self.tree_.fit()

    def predict(self, X):
        if not hasattr(self, 'tree_') or self.tree_ is None:
            raise TypeError("This {}s instance is not fitted yet.".format(type(self).__name__))
        X = check_array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        res = np.zeros(X.shape[0], dtype=self.tree_.y.dtype)
        for i, e in enumerate(X):
            currNode = self.tree_
            while currNode.classifier is None:  # 节点不是叶节点
                features_k, value_v = currNode.split_features
                if e[features_k] <= currNode.features[features_k][value_v]:
                    symbol = "<="
                else:
                    symbol = ">"
                attr = "{} {} {}".format(currNode.attributes[features_k], symbol, currNode.features[features_k][value_v])
                # attr = X[currNode.attributes.index(currNode.split_features)]
                currNode = currNode.child[attr]
                # if currNode.child[attr]:
                #     currNode = currNode.child[attr]
                # else:
                #     # 若分支节点中没有对应的属性值，则该样本标记为该分支节点中样本最多的类
                #     return currNode.freq_max_y
            res[i] = currNode.classifier
        return res

if __name__ == '__main__':
    # watermelon_data = pd.read_csv("watermelon_data_2.csv")
    # watermelon_data = pd.read_csv(r"..\linear_model\water_melon.csv", encoding='gbk')
    # X, y = watermelon_data.values[:, :-1], watermelon_data.values[:,-1]
    iris = load_iris()
    # X = iris.data[50:, :2]
    # y = iris.target[50:]
    X = iris.data[:, :2]
    # X, index = np.unique(X, axis=0, return_index=True)
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std
    y = iris.target
    # # a = NodeByID3(X, y, attributes=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    # # a.fit()
    # # a = NodeByC4_dot_5(X, y, attributes=['密度', '含糖率'])
    # # a.fit()
    # classifier1 = DecisionTreeClassifier(criterion='C4.5', attributes=['index', 'sepal_length', 'sepal_width'])
    # classifier1.fit(X, y)
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # Z = classifier1.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # # index = (Z == '是')
    # # Z[index] = 1
    # # Z[~index] = 0
    # # Z.astype('int')
    # cs = plt.contourf(xx, yy, Z, alpha=0.5)
    # plt.axis('tight')
    # colors = [[127 / 255, 127 / 255, 227 / 255], [163 / 255, 1, 213 / 255], [1, 127 / 255, 127 / 255]]
    # for i, color in zip([0, 1, 2], colors):
    #     idx = np.where(y == i)
    #     plt.scatter(X[idx, 0], X[idx, 1], c=color)
    # # for i in range(6):
    # #     print(a._IV(i))
    # plt.show()
    os.environ['path'] += os.pathsep + r"D:\suyako-to-be-coder\graphviz\bin"
    # iris = load_iris()
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    # X = np.column_stack((np.arange(150), iris.data[:, :2]))
    clf = clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    # dot_data = tree.export_graphviz(clf, out_file=None,  # doctest: +SKIP
    #                                 feature_names=iris.feature_names[:2],  # doctest: +SKIP
    #                                 class_names=iris.target_names,  # doctest: +SKIP
    #                                 filled=True, rounded=True,  # doctest: +SKIP
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")
    # graph
