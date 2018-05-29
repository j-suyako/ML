from ensemble.bagging import BaggingClassifier
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from preprocessing.data import scale
from sklearn.datasets import load_iris
from tree.tree import DecisionTreeClassifier

# define the base estimator and bagging classifier
base_classifier = DecisionTreeClassifier(criterion='C4.5', max_depth=1)
classifier1 = BaggingClassifier(base_estimator=base_classifier, n_estimators=5)


# define functions to decide which data to be estimated
def get_iris():
    iris_data = load_iris()
    X, y = iris_data.data[:, :2], iris_data.target
    X, index = np.unique(X, axis=0, return_index=True)
    X = scale(X)
    y = y[index]
    return X, y


def get_watermelon():
    watermelon_data = pd.read_csv(r"water_melon.csv", encoding='gbk')
    X = watermelon_data.values[:, :-1].astype('float')
    y = (watermelon_data.values[:, -1] == '是').astype('int')
    X = scale(X)
    return X, y


def decide_data():
    No = int(input("iris or watermelon? iris[1] watermelon[2]: "))
    if No == 1:
        return get_iris()
    elif No == 2:
        return get_watermelon()
    else:
        raise ValueError()


# get min of x, y and max of x, y in diagram
X, y = decide_data()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# scatter diagram of samples
colors = [[127 / 255, 127 / 255, 227 / 255], [163 / 255, 1, 213 / 255], [1, 127 / 255, 127 / 255]]
for classes, color in zip(np.unique(y), colors):
    idx = np.where(y == classes)
    plt.scatter(X[idx, 0], X[idx, 1], c=color)

# fit sample and plot the decision boundary
classifier1.fit(X, y)
for estimator in classifier1.estimators_:
    loc = estimator.tree_.optimal_decision_feature
    feature_k, value_v = loc
    value = estimator.tree_.feature_values[feature_k][value_v]
    if feature_k:
        plt.plot([x_min, x_max], [value, value], color='r')
    else:
        plt.plot([value, value], [y_min, y_max], color='r')

# show the diagram
plt.show()
