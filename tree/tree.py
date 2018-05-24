from abc import ABCMeta, abstractmethod
import collections
import numpy as np
from queue import Queue
from utils.validation import check_array, Check_X_y, _num_samples


def _median(X):
    """generate the median of each each neighbor value in X

    remember sort X first
    :param X: array like shape
    """
    n_samples = _num_samples(X)
    if n_samples == 1:
        yield X[0]
    for i in range(n_samples - 1):
        yield (X[i] + X[i + 1]) / 2


class Node(object):
    __metaclass__ = ABCMeta

    def __init__(self, X, y, depth=0):
        """Node in a decision tree.

        :param X: X in this node
        :param y: corresponding y in this node
        :param depth: depth of this node in the decision tree

        Attributes:
        -----------
        n_samples: int,
            sample number of X

        freq_max_y: y object,
            the class occurs most in classes, which'll be used in predict func

        classes_: y object,
            the class this node belonged to, set to None if there's two or more classes in y

        feature_values: list,
            this attribute is the list of all values of all features in X, the form may like:
            [[a1, a2], [b1, b2], ...] in ID3 node, where a1 and a2 is the feature values in
            feature a, b1, b2 are the feature values in feature b.
            In C4.5 or GINI node, the form may like: [[(a1+a2)/2, (a2+a3)/2], [(b1+b2)/2, (b2+b3)/2], ...],
            here a1,a2,b1,b2 has the same meaning in ID3 node, due to the binary attribute of C4.5 or
            GINI tree, we choose the median of near values to split current node.

        loc: list,
            loc is a corresponding attribute to attribute feature_values, which records the index of
            each value in it. For example, a feature_values like [[a1, a2], [b1, b2], ...] may create
            a loc like [(0,1),(0,2),(1,1),(1,2)], where (0,1)->a1,(0,2)->a2, etc.

        optimal_decision_feature: tuple,
            if it's a ID3 node, it'll be a tuple like (k, ), which means it uses the kth feature of
            X to split current node
            if it's a C4.5 or GINI node, it'll be a tuple like (k, v), which means it

        children: dict,
            children of this node when "divide" is performed
        """
        # X, y = Check_X_y(X, y)
        self.X = X
        self.y = y
        self.depth = depth
        n_samples = X.shape[0]
        self.n_samples = n_samples
        classes = np.unique(y)
        self.freq_max_y = classes[0] if len(classes) > 0 else classes
        self.classes_ = classes[0] if len(classes) == 1 else None
        self.optimal_decision_feature = None
        self.feature_values = list()
        self.loc = list()
        self.criterion = ""

    def ent(self):
        """entropy of samples in this node

        entropy is defined as Ent(D) = -sum(p_k * log2(p_k)), where p_k is ratio of the kth sample
        contained in all samples.
        :return:
        """
        classes_and_counts = np.unique(self.y, return_counts=True)
        entropy_of_sample = 0
        for each in classes_and_counts[1]:
            p_each = each / len(self.y)
            entropy_of_sample -= p_each * np.log2(p_each)
        return entropy_of_sample

    def _feature_choose(self, loc) -> collections.Iterable:
        """choose the index that loc means, and divide current node by self.features[index]

        if we can divide current node by self.features[index] into m subsets, then each loop
        in outer for loop will create one of these subsets.

        see loc in docstrings of __init__ method.
        """
        if hasattr(loc, "__len__"):
            features_k, value_v = loc
            if features_k < 0 or features_k >= len(self.feature_values):
                raise ValueError("No such feature_values.")
            if value_v < 0 or value_v >= len(self.feature_values[features_k]):
                raise ValueError("No such value in that feature_values.")
            for i in range(2):
                index = (self.X[:, features_k] <= self.feature_values[features_k][value_v])
                if not i:
                    X = self.X[index]
                    y = self.y[index]
                    attr = "X[{}] <= {}".format(features_k, self.feature_values[features_k][value_v])
                else:
                    X = self.X[~index]
                    y = self.y[~index]
                    attr = "X[{}] > {}".format(features_k, self.feature_values[features_k][value_v])
                yield attr, self.__class__(X, y, self.depth + 1)
        else:
            if loc < 0 or loc >= len(self.feature_values):
                raise ValueError("No such feature_values.")
            for attr in self.feature_values[loc]:
                attr = "X[{}] = {}".format(loc, attr)
                index = (self.X[:, loc] == attr)
                X = self.X[index]  # 找到每个属性值包含的训练集
                y = self.y[index]  # 该属性值包含的训练集对应的类别
                yield attr, self.__class__(X, y, self.depth + 1)

    def gain(self, loc):
        """entropy gain of current node

        this func return the entropy gain of current node if we divide it by self.features[index],
        where index is represented by loc, see loc in docstrings of __init__ method
        """
        res = self.ent()
        for _, each in self._feature_choose(loc):
            res -= each.n_samples / self.n_samples * each.ent()
        return res

    @abstractmethod
    def decision_basis(self, loc) -> float:
        """strategy about how to divide a node

        it should be overloaded in child class.

        In ID3 tree, it's information gain

        In C4.5 tree, it's information gain ratio

        In GINI tree, it's gini index
        """
        pass

    def get_best_split_feature(self):
        """get the optimal feature index

        The code could be displayed in mathematical form like this:
        arg max decision_basis(loc)

        see loc in docstring of __init__ method
        """
        if self.classes_ is not None:
            raise ValueError()
        index = np.argmax(list(map(self.decision_basis, self.loc)))
        return self.loc[index]  # 选择最优属性

    def depth_first_fit(self):
        """depth first search. Using a stack to store the tree node to avoid overflow

        The specific step could be listed as:
        1. push the root node to stack;
        2. while the stack is not empty, pop the tree node up, here we use currNode
        to label the node pop up;
        3. find the optimal division attribute of currNode;
        4. divide currNode by the optimal division attribute, also we could get the
        childNode of the currNode when we divide
        5. push the childNode to stack, and go to step 2;
        """
        DFS = list()
        DFS.append(self)
        while len(DFS) > 0:
            currNode = DFS.pop()
            if currNode.classifier is None:
                best_feature = currNode.get_best_split_feature()
                currNode.divide(best_feature)
                for child in currNode.children:
                    DFS.append(currNode.children[child])

    def breadth_first_fit(self, max_depth):
        """breadth first search, used when specific max_depth or max_leafNode(in future) has been set.

        The specific step could be listed as:
        1. enqueue the root node to queue;
        2. while the queue is not empty or depth less than the max_depth, dequeue
        the tree node, here we use currNode to label the node dequeued;
        3. find the optimal division attribute of currNode;
        4. divide currNode by the optimal division attribute, also we could get the
        childNode of the currNode when we divide;
        5. enqueue the childNode to queue, and go to step 2;
        """
        BFS = Queue()
        BFS.put(self)
        depth = 0
        while depth < max_depth and not BFS.empty():
            currNode = BFS.get()
            if currNode.classifier is None:
                best_features = currNode.get_best_split_feature()
                currNode.divide(best_features)
                for child in currNode.children:
                    BFS.put(currNode.children[child])
            depth = currNode.depth

    def divide(self, loc):
        """divide current node by self.features[index], where index is represented by loc

        see loc in docstrings of __init__ method.
        """
        features_k, value_v = loc
        self.optimal_decision_feature = (features_k, self.feature_values[features_k][value_v])
        # (features_k, value_v) = (loc[0], loc[1]) if hasattr(loc, "__len__") else (-1, -1)
        self.children = dict()
        for i, (attr, each) in enumerate(self._feature_choose(loc)):
            self.children[attr] = each

    def fit(self, max_depth=None):
        """fit root node

        choose which method(BFS or DFS) depends on whether max depth is None
        """
        if max_depth is not None:
            self.breadth_first_fit(max_depth)
        else:
            self.depth_first_fit()

    def __repr__(self):
        if hasattr(self.optimal_decision_feature, "__len__"):
            features_k, value_v = self.optimal_decision_feature
            info = ["X[{}] <= {}"]
            toshow = [(features_k, self.feature_values[features_k][value_v])]
        else:
            info = [{}]
            toshow = [self.optimal_decision_feature]
        info.extend(['{} = {}', 'samples = {}', 'classes_ = {}'])
        toshow.extend([(self.criterion, self.decision_basis(self.optimal_decision_feature)), (self.n_samples,),
                       (self.classes_,)])
        for i in range(4):
            info[i] = info[i].format(*toshow[i])
        return '[' + '\n'.join(info) + ']'

    def __str__(self):
        return self.__repr__()


class NodeByID3(Node):
    """decision tree model based on ID3

    only adapted to discrete feature
    """

    def __init__(self, X, y, depth=0):
        super(NodeByID3, self).__init__(X, y, depth=depth)
        self.criterion = "entropy"
        n_samples, n_features = X.shape
        for i in range(n_features):
            self.feature_values.append(np.unique(X[:, i]))
            self.loc.append(i)

    def decision_basis(self, loc):
        return self.gain(loc)


class NodeByC4_dot_5(Node):
    """decision tree model based on C4.5

    adapted to numerical features
    """
    def __init__(self, X, y, depth=0):
        super(NodeByC4_dot_5, self).__init__(X, y, depth=depth)
        self.criterion = "gain ratio"
        n_samples, n_features = X.shape
        for i in range(n_features):
            values = np.unique(X[:, i])
            feature_i = list(_median(values))
            self.feature_values.append(feature_i)
            p = [i] * len(feature_i)
            q = range(len(feature_i))
            self.loc.extend(list(zip(p, q)))

    def _IV(self, features_k):
        """intrinsic value of feature k

        the mathematical form is IV(fea_k) = -sum(freq(Dv) * log2(freq(Dv))), where Dv is the subset of
        samples whose feature[k] = feature[k][v], freq(Dv) means the frequency of Dv

        In general, if there were more values in feature k, IV would be greater, so C4.5 tree based on
        info gain ratio improve the tendency that ID3 tree prefer to features that has more values.
        """
        n_samples, _ = self.X.shape
        _, value_samples = np.unique(self.X[:, features_k], return_counts=True)
        value_samples_freq = value_samples / n_samples
        return -np.sum(value_samples_freq * np.log2(value_samples_freq))

    def decision_basis(self, loc):
        features_k, value_v = loc
        gain = self.gain(loc)
        IV = self._IV(features_k)
        if not gain or not IV:
            return 0
        return gain / IV


class NodeByGini(NodeByC4_dot_5):
    def __init__(self, X, y, depth=0):
        super(NodeByGini, self).__init__(X, y, depth=depth)
        self.criterion = 'gini'

    def gini(self):
        classifiers, count = np.unique(self.y, return_counts=True)
        return 1 - np.sum(np.power(count / self.n_samples, 2))

    def decision_basis(self, loc):
        res = 0
        for _, each in self._feature_choose(loc):
            res -= each.n_samples / self.n_samples * each.gini()
        return res


class DecisionTreeClassifier(object):

    def __init__(self, criterion='ID3', attributes=None, max_depth=None):
        if criterion.lower() not in ['id3', 'c4.5', 'gini']:
            raise ValueError()
        self.criterion = criterion
        self.attributes = attributes
        self.max_depth = max_depth

    def fit(self, X, y):
        if self.criterion.upper() == 'ID3':
            self.tree_ = NodeByID3(X, y)
        elif self.criterion.upper() == 'C4.5':
            self.tree_ = NodeByC4_dot_5(X, y)
        elif self.criterion.upper() == 'GINI':
            self.tree_ = NodeByGini(X, y)
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
            while currNode.classes_ is None and currNode.depth < max_depth:  # current node is not the leaf node
                features_k, value = currNode.optimal_decision_feature
                symbol = "<=" if e[features_k] <= value else ">"
                attr = "X[{}] {} {}".format(features_k, symbol, value)
                currNode = currNode.children[attr]
            res[i] = currNode.classes_ if currNode.classes_ else currNode.freq_max_y
        return res
