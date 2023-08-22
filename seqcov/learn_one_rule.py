import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

class LearnOneRule:
    def __init__(self, data, max_depth, min_samples_leaf):
        self.data = data
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        X, y = self.data.iloc[:, :-1], pd.DataFrame(self.data.iloc[:, -1])
        self.X_encoded = pd.get_dummies(X)

        self.feature_names, self.class_names = self.X_encoded.columns, self.data['Preoperative Diagnosis'].unique().astype(str)

        self.clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.clf = self.clf.fit(self.X_encoded.to_numpy(), y.to_numpy())

    def plot_classifier(self):
        plt.figure(figsize=(19, 11))
        tree.plot_tree(self.clf, filled=True, feature_names=self.feature_names, class_names=self.class_names)

    # Finds the decision tree's node with the minimum gini value
    # Returns the node's gini value, the node's decision path (list),
    # the predicted class and the number of covered samples
    def learn_one_rule(self, id, cond_feature, cond_is_right, rules):
        id_left, id_right = self.clf.tree_.children_left[id], self.clf.tree_.children_right[id]
        feature = self.clf.tree_.feature[id]

        if cond_feature is not None:
            rules.append((cond_feature, cond_is_right))

        # Leaf node
        if id_left == -1 and id_right == -1:
            # print(id, self.clf.tree_.impurity[id], rules, self.clf.tree_.value[id], np.argmax(self.clf.tree_.value[id]), self.clf.tree_.n_node_samples[id])
            return [self.clf.tree_.impurity[id], rules, np.argmax(self.clf.tree_.value[id]), self.clf.tree_.n_node_samples[id]]

        # Propagate to left and right nodes
        mgn_left, rules_left, class_left, covered_left = self.learn_one_rule(self.clf.tree_.children_left[id], self.feature_names[feature], 0, rules.copy())
        mgn_right, rules_right, class_right, covered_right = self.learn_one_rule(self.clf.tree_.children_right[id], self.feature_names[feature], 1, rules.copy())

        if mgn_left < mgn_right or (mgn_left == mgn_right and covered_left >= covered_right):
            return [mgn_left, rules_left, class_left, covered_left]
        else:
            return [mgn_right, rules_right, class_right, covered_right]
