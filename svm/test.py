import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from preprocessing.data import scale
from sklearn.datasets import load_iris
from sklearn import svm
from svm.classes import SVC

fig, ax = plt.subplots()
datasets = load_iris()
X = datasets['data'][:100, :2]
X = scale(X)
y = datasets['target'][:100]
y[np.where(y == 0)] = -1

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# plt.show()

C = 10
episilon = 0.001
classifier1 = SVC(C=C)
classifier1.fit(X, y)
# classifier2 = svm.SVC(kernel='linear', C=1)
# classifier2.fit(X, y)
# index = np.where(classifier1.alphas != 0.1 and classifier1.alphas != 0)
# w1 * x + w2 * y - b = 0
# y = -w1 / w2 * x + b / w2
coef = classifier1.w
intercept = classifier1.b
# coef = classifier2.dual_coef_
support_vector = list()
for i, alpha in enumerate(classifier1.alphas):
    if episilon < alpha < C - episilon:
        support_vector.append(plt.Circle((X[i, 0], X[i, 1]), 0.1, color='r', fill=False))
# for i, c in enumerate(coef):
for vec in support_vector:
    ax.add_artist(vec)
y1 = (-coef[0] * x_min + intercept) / coef[1]
y2 = (-coef[0] * x_max + intercept) / coef[1]
plt.plot([x_min, x_max], [y1, y2], ls='--')
plt.plot([x_min, x_max], [y1, y2], ls='--')
    # plt.show()
# print(-c[0] / c[1], intercept[i] / c[1])
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
# plt.axis('equal')
plt.show()
# print(classifier1.w)