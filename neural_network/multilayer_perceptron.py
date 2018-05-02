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

        如果我们称某连接的权重为w(j_kl)，可以理解为第j层网络上，
        输入层第k个节点与输出层第l个节点的连接的权重。

        对于最后一层隐层与输出层的
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

