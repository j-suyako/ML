from abc import ABCMeta, abstractmethod
import collections
import copy
import numpy as np
from queue import Queue
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

    def __init__(self, X, y, depth=0, max_depth=None):
        # X, y = Check_X_y(X, y)
        self.X = X  # 每个节点对应的训练集
        self.y = y  # 训练集对应的样本分类
        self.depth = depth  # 每个节点在树中的高度，初始化为0
        self.max_depth = max_depth
        n_samples = X.shape[0]
        self.n_samples = n_samples  # 训练集数量
        classifiers, count = np.unique(y, return_counts=True)
        self.freq_max_y = classifiers[0] if len(classifiers) > 0 else classifiers  # 该节点中样本最多的类，主要在predict中使用
        self.classifier = classifiers[0] if len(classifiers) == 1 else None  # 该节点所对应的类别，如果y中有不止一个分类则返回None
        self.children = {}  # 训练集的子节点，由split动态生成
        self.split_features = None
        self.features = list()
        self.feature_chosen = list()
        self.loc = list()
        self.criterion = ""

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

    def _feature_choose(self, loc) -> collections.Iterable:
        """选择X的第k个属性进行分割，生成Dv

        假设根据第k个属性，原样本可以被划分为m个集合，则每次for循环生成
        其中一个叶节点集合
        :param loc:
        :return:
        """
        if hasattr(loc, "__len__"):
            features_k, value_v = loc
            if features_k < 0 or features_k >= len(self.features):
                raise ValueError("No such features.")
            if value_v < 0 or value_v >= len(self.features[features_k]):
                raise ValueError("No such value in that features.")
            for i in range(2):
                index = (self.X[:, features_k] <= self.features[features_k][value_v])
                if not i:
                    X = self.X[index]
                    y = self.y[index]
                    attr = "X[{}] <= {}".format(features_k, self.features[features_k][value_v])
                else:
                    X = self.X[~index]
                    y = self.y[~index]
                    attr = "X[{}] > {}".format(features_k, self.features[features_k][value_v])
                yield attr, NodeByC4_dot_5(X, y, self.depth + 1, self.max_depth)
        else:
            if loc < 0 or loc >= len(self.features):
                raise ValueError("No such features.")
            for attr in self.features[loc]:
                attr = "X[{}] = {}".format(loc, attr)
                index = (self.X[:, loc] == attr)
                X = self.X[index]  # 找到每个属性值包含的训练集
                y = self.y[index]  # 该属性值包含的训练集对应的类别
                yield attr, NodeByID3(X, y, self.depth + 1, self.max_depth)

    def gain(self, loc):
        res = self.ent()
        for _, each in self._feature_choose(loc):
            res -= each.n_samples / self.n_samples * each.ent()
        return res

    @abstractmethod
    def decision_basis(self, loc) -> float:
        pass

    def get_best_split_feature(self):
        if self.classifier is not None:
            raise ValueError()
        index, _ = max(enumerate(map(self.decision_basis, self.loc)), key=lambda p: p[1])
        return self.loc[index]  # 选择最优属性

    def depth_first_fit(self):
        DFS = list()
        DFS.append(self)
        while len(DFS) > 0:
            currNode = DFS.pop()
            if currNode.classifier is None:
                best_features = currNode.get_best_split_feature()
                currNode.set_split_features(best_features)
                for child in currNode.children:
                    DFS.append(currNode.children[child])

    def breadth_first_fit(self):
        BFS = Queue()
        BFS.put(self)
        depth = 0
        while depth < self.max_depth and not BFS.empty():
            currNode = BFS.get()
            if currNode.classifier is None:
                best_features = currNode.get_best_split_feature()  # 选择最优属性
                currNode.set_split_features(best_features)  # 用该最优属性进行划分
                for child in currNode.children:
                    BFS.put(currNode.children[child])
            depth = currNode.depth

    def set_split_features(self, loc):
        self.split_features = loc
        (features_k, value_v) = (loc[0], loc[1]) if hasattr(loc, "__len__") else (-1, -1)
        for i, (attr, each) in enumerate(self._feature_choose(loc)):
            self.children[attr] = each
            self.children[attr].feature_chosen = copy.deepcopy(self.feature_chosen)
            if features_k > -1:
                while 0 <= value_v < len(self.feature_chosen[features_k]):
                    self.children[attr].feature_chosen[features_k][value_v] = True
                    value_v += 1 if not i else -1
            else:
                self.children[attr].feature_chosen[loc] = True

    def fit(self):
        if self.max_depth is not None:
            self.breadth_first_fit()
        else:
            self.depth_first_fit()

    def __repr__(self):
        if hasattr(self.split_features, "__len__"):
            features_k, value_v = self.split_features
            info = ["X[{}] <= {}"]
            toshow = [(features_k, self.features[features_k][value_v])]
        else:
            info = [{}]
            toshow = [self.split_features]
        info.extend(['{} = {}', 'samples = {}', 'classifier = {}'])
        toshow.extend([(self.criterion, self.decision_basis(self.split_features)), (self.n_samples, ),
                       (self.classifier, )])
        for i in range(4):
            info[i] = info[i].format(*toshow[i])
        return '[' + '\n'.join(info) + ']'

    def __str__(self):
        return self.__repr__()


class NodeByID3(Node):
    """基于ID3决策树算法的模型

    最优属性划分采用的是信息熵增益的方法（注意与C4.5信息熵增益率的区别）
    只能用于非连续属性的划分（实际上二分法对这个应该没有限制，我也不懂为什么都说ID3不能处理连续值）
    """

    def __init__(self, X, y, depth=0, max_depth=None):
        super(NodeByID3, self).__init__(X, y, depth=depth, max_depth=max_depth)
        self.criterion = "entropy"
        n_samples, n_features = X.shape
        for i in range(n_features):
            self.features.append(np.unique(X[:, i]))
            self.feature_chosen.append(False)
            self.loc.append(i)

    def decision_basis(self, loc):
        if self.feature_chosen[loc]:
            return 0
        return self.gain(loc)


class NodeByC4_dot_5(Node):
    """基于C4.5决策树算法的模型

    最优属性划分采用信息熵增益率的方法（注意与ID3信息熵增益的区别）
    可以处理连续变量
    """
    def __init__(self, X, y, depth=0, max_depth=None):
        super(NodeByC4_dot_5, self).__init__(X, y, depth=depth, max_depth=max_depth)
        self.criterion = "gain ratio"
        n_samples, n_features = X.shape
        for i in range(n_features):
            values = np.unique(X[:, i])
            feature_i = list(_median(values))
            self.features.append(feature_i)
            self.feature_chosen.append([False] * len(values))
            p = [i] * len(feature_i)
            q = range(len(feature_i))
            self.loc.extend(list(zip(p, q)))

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

    def decision_basis(self, loc):
        features_k, value_v = loc
        if self.feature_chosen[features_k][value_v]:
            return 0
        gain = self.gain(loc)
        IV = self._IV(features_k)
        if not gain or not IV:
            return 0
        return gain / IV


class DecisionTreeClassifier(object):

    def __init__(self, criterion='ID3', attributes=None, max_depth=None):
        self.criterion = criterion
        self.attributes = attributes
        self.max_depth = max_depth

    def fit(self, X, y):
        if self.criterion == 'ID3':
            self.tree_ = NodeByID3(X, y, max_depth=self.max_depth)
        elif self.criterion == 'C4.5':
            self.tree_ = NodeByC4_dot_5(X, y, max_depth=self.max_depth)
        self.tree_.fit()

    def predict(self, X):
        if not hasattr(self, 'tree_') or self.tree_ is None:
            raise TypeError("This {}s instance is not fitted yet.".format(type(self).__name__))
        if self.max_depth is None:
            max_depth = 1e3
        else:
            max_depth = self.max_depth
        X = check_array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        res = np.zeros(X.shape[0], dtype=self.tree_.y.dtype)
        for i, e in enumerate(X):
            currNode = self.tree_
            while currNode.classifier is None and currNode.depth < max_depth:  # 节点不是叶节点
                features_k, value_v = currNode.split_features
                symbol = "<=" if e[features_k] <= currNode.features[features_k][value_v] else ">"
                attr = "X[{}] {} {}".format(features_k, symbol, currNode.features[features_k][value_v])
                currNode = currNode.children[attr]
            res[i] = currNode.classifier if currNode.classifier else currNode.freq_max_y
        return res
