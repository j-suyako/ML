import numpy as np
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
        """初始化权重，暂时全部归零 TODO(suyako): 参考sklearn的初始化方法

        在这里返回一个shape为in x out的ndarray coef，对于(i, j)坐标来说，
        coef(i, j)为输入层的第i个节点与输出层的第j个节点连接的权重
        :param fan_in: 作为输入层的神经网络节点数
        :param fan_out: 作为输出层的神经网络节点数
        :return:
        """
        coef_ = np.zeros((fan_in, fan_out))
        return coef_

    def _backprop(self, X, y):
        """暂时只写X只有一个样本的情况，默认y已转为one hot vector

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
        lvs = self.layer_values
        for i in range(self.n_layers - 1, 0, -1):
            grads = np.zeros_like(self.coef_[i])
            if i == self.n_layers - 1:
                for h, grad in enumerate(grads):
                    # delta_w_hj = n * g_j * b_h，其中g_j = y_j_hat * (1 - y_j_hat) * (y_j - y_j_hat)
                    # 这里y_j_hat是预测值，y_j是实际值
                    grad = 0.1 * lvs[-1] * (1 - lvs[-1]) * (y - lvs[-1]) * lvs[-2][h]
            else:
                # fai_E_k / fai_V_ih = (sum(fai_E_k / fai_w_hj * w_hj) * (1 - b_h))
                pass # TODO(suyako): 实现向后传播

    def fit(self, X, y):
        """暂时只认为X只有一个样本来做

        """
        X, y = Check_X_y(X, y)
        n_samples, n_features = _num_samples(X)
        labels = len(np.unique(y))
        self.coef_ = []
        self.layer_values = [X]
        input_layer_size = n_features  # TODO(suyako): +1?
        output_layer_size = labels
        hidden_layer_sizes = list(self.hidden_layer_sizes)
        layer_units = ([input_layer_size] + hidden_layer_sizes + [output_layer_size])
        self.n_layers = len(layer_units)
        for i in range(self.n_layers - 1):
            coef_ = self._init_coef(layer_units[i], layer_units[i + 1])
            self.coef_.append(coef_)  # TODO(suyako): 暂时构建到系数矩阵
            layer_value = np.dot(coef_.T, self.layer_values[i])
            self.layer_values.append(layer_value)
        # 第一次神经网络遍历

