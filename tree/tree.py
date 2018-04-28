import numpy as np
import pandas as pd


class Node(object):
    # nodeToFit = Queue()  # 用一个队列来保存待求增益的节点

    def __init__(self, X, y, attributes=None):
        self.X = X  # 每个节点对应的训练集
        self.y = y  # 训练集对应的样本分类
        self.samples = len(y)  # 训练集数量
        self.child = {}  # 训练集的子节点，由split动态生成
        if not attributes:
            attributes = range(X.shape[1])
        self.attributes = attributes  # attributes代表属性集，如果缺省的话给一个0 - N-1的列表，其中N是X的属性集数量
        self.split_features = self.attributes[0]  # 训练集的最优划分属性
        self.classifier = np.unique(y)[0] if len(np.unique(y)) == 1 else None  # 该节点所对应的类别，如果y中有不止一个分类则返回None
        self.feature_chosen = []  # 用来保存已经用来分割过的属性集

    def ent(self):
        classes_and_counts = np.unique(self.y, return_counts=True)
        entropy_of_sample = 0
        for each in classes_and_counts[1]:
            p_each = each / len(self.y)
            entropy_of_sample -= p_each * np.log2(p_each)
        return entropy_of_sample

    def _feature_choose(self, features_k):
        '''选择X的第k个属性进行分割，生成Dv'''
        if features_k < 0 or features_k >= len(self.attributes):
            raise ValueError("No such features.")
        for each_value in np.unique(self.X[:, features_k]):
            index = (self.X[:, features_k] == each_value)
            X = self.X[index]  # 找到每个属性值包含的训练集
            y = self.y[index]  # 该属性值包含的训练集对应的类别
            yield each_value, Node(X, y, self.attributes)

    def gain(self, features_k):
        '''计算用第k个属性对样本进行划分得到的信息增益'''
        res = self.ent()
        for _, each in self._feature_choose(features_k):
            res -= each.samples / self.samples * each.ent()
        return res

    def fit(self):
        if self.classifier is None:
            Gain = list()
            Gain_index = list()
            for k, _ in enumerate(self.attributes):
                if k in self.feature_chosen:
                    continue
                Gain.append(self.gain(k))
                Gain_index.append(k)
            best_features = Gain_index[Gain.index(max(Gain))]
            self.set_split_features(best_features)
            for each_child in self.child:
                self.child[each_child].fit()
        else:
            self.split_features = None

    def set_split_features(self, features_k):
        self.split_features = self.attributes[features_k]
        for attr, each in self._feature_choose(features_k):
            self.child[attr] = each
        self.feature_chosen.append(features_k)

    def __repr__(self):
        info = ['splitter = {}', 'Ent = {}', 'samples = {}', 'class = {}']
        toshow = [self.split_features, self.ent(), self.samples, self.classifier]
        for i in range(4):
            info[i] = info[i].format(toshow[i])
        return '[' + '\n'.join(info) + ']'

    def __str__(self):
        return self.__repr__()


class DecisionTreeClassifier(object):

    def __init__(self, criterion='entropy'):
        self.criterion = criterion

    def fit(self, X, y):
        self.tree_ = Node(X, y)
        self.tree_.fit()

    def predict(self, X):
        currNode = self.tree_
        while not currNode.classifier:
            attr = X[currNode.attributes.index(currNode.split_features)]
            currNode = currNode.child[attr]
        return currNode.classifier

if __name__ == '__main__':
    watermelon_data = pd.read_csv("watermelon_data_2.csv")
    X, y = watermelon_data.values[:, :-1], watermelon_data.values[:,-1]
    a = Node(X, y, attributes=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    a.fit()
    pass