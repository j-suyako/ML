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

    def fit(self, X, y):
        X, y = Check_X_y(X, y)
        n_samples, n_features = _num_samples(X)
        labels = len(np.unique(y))
        self.coef_ = []
        input_layer_size = n_features  # TODO(suyako): +1?
        output_layer_size = labels
        hidden_layer_sizes = list(self.hidden_layer_sizes)
        layer_units = ([input_layer_size] + hidden_layer_sizes + [output_layer_size])
        self.n_layers = len(layer_units)
        for i in range(self.n_layers - 1):
            coef_ = np.zeros((layer_units[i], layer_units[i + 1]))
            self.coef_.append(coef_)  # TODO(suyako): 暂时构建到系数矩阵
