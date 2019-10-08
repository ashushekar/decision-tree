from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Before Pre-prunning.....")
print("Accuracy on Training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Test set: {:.3f}".format(tree.score(X_test, y_test)))

# Lets apply Pre-prunning to the tree, which will stop developing the tree.
# With limited depth, this will reduce overfitting
tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)
print("\nAfter Pre-prunning.....")
print("Accuracy on Training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Test set: {:.3f}".format(tree.score(X_test, y_test)))

# Feature importance: rates how important each feature is for the decison a tree makes.
# It is a number between 0 and 1 for each feature, where 0 means "not used at all" and
# 1 means "perfectly predicts the target"
print("Feature Importance of the tree is:\n{}".format(tree.feature_importances_))


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show(block=True)


plot_feature_importances_cancer(tree)