import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from ._base import logistic
from ..preprocessing.data import scale
from ..utils.validation import check_array, Check_X_y, _num_samples


class MLPClassifier(object):
    """多层感知机

    """
    def __init__(self, hidden_layer_sizes=(3,), activation='logistic',
                 solver='sgd'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver

    def _init_coef(self, fan_in, fan_out):
        """初始化权重

        在这里返回一个shape为(in + 1) x out的ndarray coef，对于(i, j)坐标来说，
        coef(i, j)为输入层的第i个节点与输出层的第j个节点连接的权重
        :param fan_in: 作为输入层的神经网络节点数
        :param fan_out: 作为输出层的神经网络节点数
        :return:
        """
        coef_ = np.random.random((fan_in + 1, fan_out))
        return coef_

    def _backprop(self, X, y):
        """反向传播，默认y已转为one hot vector（转为one hot vector则输出层上每个节点的输出值具有概率意义）

        这里给一个反向传播的初步解释：
        对第l层上的第j个节点来说，对该节点的输入值（即未激活前）一个小的扰动Delta_Z_l^j，则损失函数变化
        Delta_cost = fai_cost / fai_Z_j^l * Delta_Z_j^l
        在这里我们称fai_cost / fai_Z_j^l为该节点内含的误差delta_j^l（与常规认识的误差有概念上的区别）
        随着网络的不断前进，前一层的节点误差会不断重分布在下一层节点上，这意味着前后两层的误差是可以互推的，
        反向传播就是一种通过后一层误差反向推导出前一层误差的方法。

        显然我们的最终目标是让delta_j^l达到0，而为了让其达到0我们需要不断调整权重，所以最终目标转换为
        让损失函数对所有权重的偏导为0，损失函数对权重的偏导可由delta_j^l推导得出。设第l-1层上的第k个节点
        与第l层上的第j个节点连接的权重为w_kj^l，其引起的第l层上第j个节点输入值的变化为w_kj^l * a_k^(l-1)，
        所以最终引起的损失函数变化为delta_j^i * w_kj^l * a_k^(l-1)，所以损失函数对w_kj^l的偏导就为
        delta_j^i * a_k^(l-1)。

        更详细的解释见ref: http://neuralnetworksanddeeplearning.com/chap2.html
        :param X:
        :param y:
        :return:
        """
        if not hasattr(self, 'coef_') or not self.coef_:
            raise TypeError()
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        X = X[index]  # 为之后的批量梯度下降对X进行重排
        y = y[index]
        errs = [np.zeros(0)] * self.n_layers
        costs = list()
        for loop in range(100):
            costs.append(self._cost(X=X, y=y)) # 获得在此时的权重矩阵下神经网络的损失
            x = X[loop]  # 选取X中的一个样本，这里loop只是先写着（因为方便）
            activations = [np.r_[x, 1]]  # 考虑常量
            activations = self._forward_pass(activations)  # 获得网络上所有节点的激活值
            for i in range(self.n_layers - 1, -1, -1):
                # 依靠误差反向传播获得每层节点的误差
                if i == self.n_layers - 1:
                    errs[i] = (activations[i] - y[loop]) * activations[i] * (1 - activations[i])
                errs[i] = np.dot(self.coef_[i + 1].T, errs[i + 1]) * activations[i] * (1 - activations[i])
            for i in range(self.n_layers - 1):
                # 由每层节点的误差对权重矩阵进行更新，以下写成了矩阵形式
                left = np.tile(activations[i], (errs[i + 1].shape[0], 1)).T
                right = np.tile(errs[i + 1], (activations[i].shape[0], 1))
                grad = left * right
                self.coef_[i] -= 0.02 * grad
        # lvs = self.layer_values
        # for i in range(self.n_layers - 1, 0, -1):
        #     grads = np.zeros_like(self.coef_[i])
        #     if i == self.n_layers - 1:
        #         for h, grad in enumerate(grads):
        #             # delta_w_hj = n * g_j * b_h，其中g_j = y_j_hat * (1 - y_j_hat) * (y_j - y_j_hat)
        #             # 这里y_j_hat是预测值，y_j是实际值
        #             grad = 0.1 * lvs[-1] * (1 - lvs[-1]) * (y - lvs[-1]) * lvs[-2][h]
        #     else:
        #         # fai_E_k / fai_V_ih = (sum(fai_E_k / fai_w_hj * w_hj) * (1 - b_h))
        #         pass # TODO(suyako): 实现向后传播

    def _forward_pass(self, activations):
        """输入值在神经网络中进行传递，获得每层的激活值

        :param activations:
        :return:
        """
        for i, coef in enumerate(self.coef_):
            unactivation = np.dot(coef.T, activations[i])
            activations.append(np.r_[logistic(unactivation), 1] if i < self.n_layers - 1 else unactivation)
        return activations

    def _cost(self, X, y):
        """神经网络损失函数

        这里采用MSR，当然这里y与y_hat对调也没事
        :param y: 实际值
        :param y_hat: 预测值
        :return:
        """
        cost = 0
        for i, x in enumerate(X):
            activations = np.r_[x, 1]
            y_hat = self._forward_pass(activations)[-1]
            cost += np.sum((y_hat - y[i]) * (y_hat - y[i])) / 2
        return cost

    def fit(self, X, y):
        """暂时只认为X只有一个样本来做

        """
        X, y = Check_X_y(X, y)
        n_samples, n_features = _num_samples(X)
        labels = len(np.unique(y))
        self.coef_ = []
        # self.layer_values = [X]
        input_layer_size = n_features  # TODO(suyako): +1?
        output_layer_size = labels
        hidden_layer_sizes = list(self.hidden_layer_sizes)
        layer_units = ([input_layer_size] + hidden_layer_sizes + [output_layer_size])
        self.n_layers = len(layer_units)
        for i in range(self.n_layers - 1):
            coef_ = self._init_coef(layer_units[i], layer_units[i + 1])
            self.coef_.append(coef_)  # TODO(suyako): 暂时构建到系数矩阵
        self._backprop(X, y)  # 反向传播更新权重矩阵
            # layer_value = np.dot(coef_.T, self.layer_values[i])
            # self.layer_values.append(layer_value)
        # 第一次神经网络遍历

if __name__ == '__main__':
    X, y = make_moons(noise=0.3, random_state=0)
    figure = plt.figure(figsize=(17, 9))
    X = scale(X)
    classifier1 = MLPClassifier()
    classifier1.fit(X, y)
