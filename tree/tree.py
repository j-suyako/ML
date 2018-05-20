from abc import ABCMeta, abstractmethod
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

    @abstractmethod
    def _feature_choose(self, *args):
        pass

    def gain(self, *args):
        res = self.ent()
        for _, each in self._feature_choose(*args):
            res -= each.n_samples / self.n_samples * each.ent()
        return res

    @abstractmethod
    def get_best_split_feature(self):
        pass

    def depth_first_fit(self):
        if self.classifier is None:
            best_features = self.get_best_split_feature()
            self.set_split_features(*best_features)
            for child in self.children:
                self.children[child].fit()

    def breadth_first_fit(self):
        BFS = Queue()
        BFS.put(self)
        depth = 0
        while depth < self.max_depth and not BFS.empty():
            currNode = BFS.get()
            if currNode.classifier is None:
                best_features = currNode.get_best_split_feature()  # 选择最优属性
                currNode.set_split_features(*best_features)  # 用该最优属性进行划分
                for child in currNode.children:
                    BFS.put(currNode.children[child])
            depth = currNode.depth

    @abstractmethod
    def set_split_features(self, *args):
        pass

    def fit(self):
        if self.max_depth is not None:
            self.breadth_first_fit()
        else:
            self.depth_first_fit()


class NodeByID3(Node):
    """基于ID3决策树算法的模型

    最优属性划分采用的是信息熵增益的方法（注意与C4.5信息熵增益率的区别）
    只能用于非连续属性的划分（实际上二分法对这个应该没有限制，我也不懂为什么都说ID3不能处理连续值）
    """

    def __init__(self, X, y, depth=0, max_depth=None):
        super(NodeByID3, self).__init__(X, y, depth=depth, max_depth=max_depth)
        n_samples, n_features = X.shape
        for i in range(n_features):
            self.features.append(np.unique(X[:, i]))
            self.feature_chosen.append(False)

    def _feature_choose(self, features_k):
        """选择X的第k个属性进行分割，生成Dv

        假设根据第k个属性，原样本可以被划分为m个集合，则每次for循环生成
        其中一个叶节点集合
        :param features_k:
        :return:
        """
        if features_k < 0 or features_k >= len(self.features):
            raise ValueError("No such features.")
        for attr in self.features[features_k]:
            attr = "X[{}] = {}".format(features_k, attr)
            index = (self.X[:, features_k] == attr)
            X = self.X[index]  # 找到每个属性值包含的训练集
            y = self.y[index]  # 该属性值包含的训练集对应的类别
            yield attr, NodeByID3(X, y, self.depth + 1, self.max_depth)

    def get_best_split_feature(self):
        if self.classifier is not None:
            raise ValueError
        Gain = list()
        Gain_index = list()
        for k, _ in enumerate(self.features):
            # 如果该属性已经在之前被选择过作为划分依据，则跳过
            if self.feature_chosen[k]:
                continue
            Gain.append(self.gain(k))  # 计算每个属性划分所带来的信息增益并存入Gain
            Gain_index.append(k)  # 同时记录被用来试验的属性的index
        return Gain_index[Gain.index(max(Gain))]  # 选择最优属性

    def set_split_features(self, features_k):
        # self.split_features = self.attributes[features_k]
        self.split_features = features_k
        for attr, each in self._feature_choose(features_k):
            self.children[attr] = each  # child中保存由该最优属性划分得到的每个属性值对应的叶节点
            # self.children[attr].feature_chosen[self.feature_chosen] = True
            self.children[attr].feature_chosen = copy.copy(self.feature_chosen)
            self.children[attr].feature_chosen[features_k] = True

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
    def __init__(self, X, y, depth=0, max_depth=None):
        super(NodeByC4_dot_5, self).__init__(X, y, depth=depth, max_depth=max_depth)
        n_samples, n_features = X.shape
        for i in range(n_features):
            values = np.unique(X[:, i])
            self.features.append(list(_median(values)))
            self.feature_chosen.append([False] * len(values))

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
                attr = "X[{}] <= {}".format(features_k, self.features[features_k][value_v])
            else:
                X = self.X[~index]
                y = self.y[~index]
                attr = "X[{}] > {}".format(features_k, self.features[features_k][value_v])
            yield attr, NodeByC4_dot_5(X, y, self.depth + 1, self.max_depth)

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

    def gain_ratio(self, args):
        features_k, value_v = args
        if self.feature_chosen[features_k][value_v]:
            return 0
        gain = self.gain(*args)
        IV = self._IV(features_k)
        if not gain or not IV:
            return 0
        return gain / IV

    def get_best_split_feature(self):
        if self.classifier is not None:
            raise ValueError()
        Gain_ratio_index = list()
        for i, feature in enumerate(self.features):
            p = [i] * len(feature)
            q = range(len(feature))
            Gain_ratio_index.extend(list(zip(p, q)))
        index, _ = max(enumerate(map(self.gain_ratio, Gain_ratio_index)), key=lambda p: p[1])
        return Gain_ratio_index[index]

    def set_split_features(self, features_k, value_v):
        self.split_features = (features_k, value_v)
        # self.split_features = self.features[features_k][value_v]
        for i, (attr, each) in enumerate(self._feature_choose(features_k, value_v)):
            self.children[attr] = each
            self.children[attr].feature_chosen = copy.deepcopy(self.feature_chosen)
            while 0 <= value_v < len(self.feature_chosen[features_k]):
                self.children[attr].feature_chosen[features_k][value_v] = True
                if not i:
                    value_v += 1
                else:
                    value_v -= 1


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
