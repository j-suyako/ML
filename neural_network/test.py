import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from neural_network.multilayer_perceptron import MLPClassifier
# from sklearn.neural_network import MLPClassifier
import numpy as np
from preprocessing.data import scale
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.datasets import load_digits

if __name__ == '__main__':
    X, y = make_moons(noise=0.3, random_state=0)
    # X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
    # X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
    #                            random_state=0, n_clusters_per_class=1)
    # rng = np.random.RandomState(2)
    # X += 2 * rng.uniform(size=X.shape)
    # linearly_separable = (X, y)
    X = scale(X)
    # X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    # y = [1, 0, 1, 0]
    # digits = load_digits()
    # n_samples = len(digits.images)
    # data = digits.images.reshape((n_samples, -1))
    # X, y = data[:n_samples//2], digits.target[:n_samples//2]

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # plt.show()
    classifier1 = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=500)
    classifier1.fit(X, y)
    Z = classifier1.predict(np.c_[np.ravel(xx), np.ravel(yy)])
    # Z = classifier1.predict(data[n_samples//2:])
    # temp = Z - digits.target[n_samples//2:]
    # pass
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.show()