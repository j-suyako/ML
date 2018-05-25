import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from queue import Queue
from sklearn.datasets import load_iris
from sklearn import tree
import sys
from tree import DecisionTreeClassifier

# watermelon_data = pd.read_csv("watermelon_data_2.csv")
# watermelon_data = pd.read_csv(r"..\linear_model\water_melon.csv", encoding='gbk')
# X, y = watermelon_data.values[:, :-1], watermelon_data.values[:,-1]
iris = load_iris()
# X = iris.data[50:, :2]
# y = iris.target[50:]
X = iris.data[:, :2]
X, index = np.unique(X, axis=0, return_index=True)
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
y = iris.target[index]
# # a = NodeByID3(X, y, attributes=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
# # a.fit()
# # a = NodeByC4_dot_5(X, y, attributes=['密度', '含糖率'])
# # a.fit()
classifier1 = DecisionTreeClassifier(criterion='GINI')#, max_depth=4)
classifier1.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = classifier1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# index = (Z == '是')
# Z[index] = 1
# Z[~index] = 0
# Z.astype('int')
cs = plt.contourf(xx, yy, Z, alpha=0.5)
plt.axis('tight')
colors = [[127 / 255, 127 / 255, 227 / 255], [163 / 255, 1, 213 / 255], [1, 127 / 255, 127 / 255]]
for i, color in zip([0, 1, 2], colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color)
# for i in range(6):
#     print(a._IV(i))
plt.show()
# os.environ['path'] += os.pathsep + r"D:\suyako-to-be-coder\graphviz\bin"
# # iris = load_iris()
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# # X = np.column_stack((np.arange(150), iris.data[:, :2]))
# clf = clf.fit(X, y)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
# dot_data = tree.export_graphviz(clf, out_file=None,  # doctest: +SKIP
#                                 feature_names=iris.feature_names[:2],  # doctest: +SKIP
#                                 class_names=iris.target_names,  # doctest: +SKIP
#                                 filled=True, rounded=True,  # doctest: +SKIP
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
# graph