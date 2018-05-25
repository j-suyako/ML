from ensemble.bagging import BaggingClassifier
from sklearn.datasets import load_iris
from tree.tree import DecisionTreeClassifier

base_classifier = DecisionTreeClassifier(criterion='C4.5')
classifier1 = BaggingClassifier(base_estimator=base_classifier, n_estimators=3)
iris_data = load_iris()
X = iris_data.values[:2, :]
y = iris_data.target
classifier1.fit(X, y)