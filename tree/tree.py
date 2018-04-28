from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

# 所有决策树算法暂时不考虑连续变量，sklearn采用的应该是所有属性当连续变量处理然后二分。

class Node(object):
    __metaclass__ = ABCMeta

    def __init__(self, X, y, attributes=None):
        self.X = X  # 每个节点对应的训练集
        self.y = y  # 训练集对应的样本分类
        classifiers, count = np.unique(y, return_counts=True)
        self.freq_max_y = classifiers[0]  # 该节点中样本最多的类，主要在predict中使用
        self.n_samples = len(y)  # 训练集数量
        self.child = {}  # 训练集的子节点，由split动态生成
        if not attributes:
            attributes = range(X.shape[1])
        self.attributes = attributes  # attributes代表属性集，如果缺省的话给一个0 - N-1的列表，其中N是X的属性集数量
        self.split_features = None
        # self.split_features = self.attributes[0]  # 训练集的最优划分属性
        self.classifier = classifiers[0] if len(classifiers) == 1 else None  # 该节点所对应的类别，如果y中有不止一个分类则返回None
        self.feature_chosen = []  # 用来保存已经用来分割过的属性集

    @abstractmethod
    def fit(self):
        pass


class NodeByID3(Node):
    # nodeToFit = Queue()  # 用一个队列来保存待求增益的节点

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
        if features_k < 0 or features_k >= len(self.attributes):
            raise ValueError("No such features.")
        for each_value in np.unique(self.X[:, features_k]):
            index = (self.X[:, features_k] == each_value)
            X = self.X[index]  # 找到每个属性值包含的训练集
            y = self.y[index]  # 该属性值包含的训练集对应的类别
            yield each_value, NodeByID3(X, y, self.attributes)

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
            for k, _ in enumerate(self.attributes):
                # 如果该属性已经在之前被选择过作为划分依据，则跳过
                if k in self.feature_chosen:
                    continue
                Gain.append(self.gain(k))  # 计算每个属性划分所带来的信息增益并存入Gain
                Gain_index.append(k)  # 同时记录被用来试验的属性的index
            best_features = Gain_index[Gain.index(max(Gain))]  # 选择最优属性
            self.set_split_features(best_features)  # 用该最优属性进行划分
            # 在已选择属性向量中增加该最优属性（Attention！每个分支节点对应的已选择属性都是不一样的）
            self.feature_chosen.append(best_features)
            for each_child in self.child:
                self.child[each_child].fit()  # 递归
        # else:
        #     self.split_features = None

    def set_split_features(self, features_k):
        self.split_features = self.attributes[features_k]
        for attr, each in self._feature_choose(features_k):
            self.child[attr] = each  # child中保存由该最优属性划分得到的每个属性值对应的叶节点

    def __repr__(self):
        info = ['splitter = {}', 'Ent = {}', 'n_samples = {}', 'class = {}']
        toshow = [self.split_features, self.ent(), self.n_samples, self.classifier]
        for i in range(4):
            info[i] = info[i].format(toshow[i])
        return '[' + '\n'.join(info) + ']'

    def __str__(self):
        return self.__repr__()


class NodeByC4_dot_5(Node):
    def __init__(self, X, y, attributes=None):
        super(NodeByC4_dot_5, self).__init__(X, y, attributes=attributes)


class DecisionTreeClassifier(object):

    def __init__(self, criterion='entropy'):
        self.criterion = criterion

    def fit(self, X, y):
        self.tree_ = NodeByID3(X, y)
        self.tree_.fit()

    def predict(self, X):
        if not hasattr(self, 'tree_') or self.tree_ is None:
            raise TypeError("This {}s instance is not fitted yet.".format(type(self).__name__))
        currNode = self.tree_
        while not currNode.classifier:  # 节点不是叶节点
            attr = X[currNode.attributes.index(currNode.split_features)]
            if currNode.child[attr]:
                currNode = currNode.child[attr]
            else:
                # 若分支节点中没有对应的属性值，则该样本标记为该分支节点中样本最多的类
                return currNode.freq_max_y
        return currNode.classifier

if __name__ == '__main__':
    watermelon_data = pd.read_csv("watermelon_data_2.csv")
    X, y = watermelon_data.values[:, :-1], watermelon_data.values[:,-1]
    a = NodeByID3(X, y, attributes=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    a.fit()
    pass