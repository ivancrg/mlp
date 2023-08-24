import pandas as pd
from sklearn import tree
from learn_one_rule import LearnOneRule
import matplotlib.pyplot as plt


class SequentialCovering:

    def __init__(self, data, multiclass=False, max_depth=3, min_samples_leaf=1):
        self.data_orig = data
        self.data_orig.reset_index(drop=True, inplace=True)
        self.multiclass = multiclass
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.result = None

    def remove_covered_instances(self, data, rule):
        data.reset_index(drop=True, inplace=True)

        # print(f'Data length: {len(data)}')

        # Creating condition for removing instances
        # (all instances under remove condition should be removed)
        remove_condition = pd.Series([True] * len(data))
        for condition in rule:
            feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

            if op == '<=':
                remove_condition &= (data[feat] <= thr)
            elif op == '>':
                remove_condition &= (data[feat] > thr)
            else:
                print("main.py::remove_covered_instances WARNING: Unknown operator!")

        # Filtering instances which should be removed (because they're covered by the new_rule)
        new_data = data[~remove_condition]

        return new_data

    def fit(self):
        if self.multiclass:
            self.result = self.sc_multiclass(self.data_orig.copy())
        else:
            self.result = self.sc(self.data_orig.copy())

    def sc(self, data):
        rules_preds = []

        while len(data) > 0:
            data.reset_index(drop=True, inplace=True)

            lor = LearnOneRule(data.copy(), max_depth=self.max_depth,
                               min_samples_leaf=self.min_samples_leaf)
            _, new_rule, pred, n_covered = lor.learn_one_rule(
                0, None, None, None, [])
            # lor.plot_classifier()
            # plt.show()

            # Append a rule as a tuple of an array of (feature, value) conditions and outcome
            rules_preds.append((new_rule, pred))

            # Removing covered instances
            data = self.remove_covered_instances(data, new_rule)

        return rules_preds

    def sc_multiclass(self, data):
        # Array of tuples - (class, number of instances of that class)
        y = pd.DataFrame(data.iloc[:, -1])
        classes_counts = [(c, sum(y.iloc[:, 0] == c))
                          for c in y.iloc[:, 0].unique()]
        classes_counts.sort(key=lambda x: x[1])

        rules = []

        # print(classes_counts, y.value_counts(ascending=True))

        # While there are classes for which the rule hasn't been set yet
        while len(classes_counts) > 1:
            data.reset_index(drop=True, inplace=True)

            # Current class for which a rule should be set
            current_class = classes_counts[0][0]

            # print(f'Current class: {current_class}')

            # Copy of the dataset to be modified to binary classification
            data_current = data.copy()
            data_current.loc[data.iloc[:, -1] ==
                             current_class, 'Preoperative Diagnosis'] = 1
            data_current.loc[data.iloc[:, -1] !=
                             current_class, 'Preoperative Diagnosis'] = 0

            # Calculating the rule for the current class
            rules_preds_bin = self.sc(data_current)

            # Remove all covered instances using rules_binary
            for r, pred in rules_preds_bin:
                # We remove only positive data instances
                if pred == 1:
                    data = self.remove_covered_instances(data, r)

            # Add the new rules to result
            rules.append(
                {'class': current_class, 'rules_preds': rules_preds_bin})

            # Removing the class for which the rule has been calculated
            classes_counts.remove(classes_counts[0])

        # Adding a default rule
        rules.append({
            'class': classes_counts[0][0],
            'rules_preds': [([{
                'feature': 'default',
                'operator': '',
                'threshold': ''
            }], 1)]})

        return rules

    def predict(self, input):
        if self.multiclass:
            return self.predict_mc(self.result, input)
        else:
            return self.predict_binary(self.result, input)

    def predict_binary(self, result, input):
        input.loc[:, 'Prediction'] = "N/A"

        for rule_pred in result:
            rule, pred = rule_pred

            # Keeps the record of all instances valid for
            # assigning a prediction
            pred_condition = (input['Prediction'] == "N/A")

            for condition in rule:
                feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

                if op == '<=':
                    pred_condition &= (input[feat] <= thr)
                elif op == '>':
                    pred_condition &= (input[feat] > thr)
                else:
                    print(
                        f"main.py::predict WARNING: Unknown operator! {feat} {op} {thr} {len(rule)}")

            input.loc[pred_condition, 'Prediction'] = pred

        return input

    def predict_mc(self, clf, input):
        input.loc[:, 'Prediction'] = "N/A"

        for c in clf:
            current_class = c['class']

            # Keeps the record of all instances valid for
            # assigning a prediction of current class
            # (e.g. if a prediction is 0, the valid values of
            #  appropriate instances will be set to False)
            valid = (input['Prediction'] == "N/A")

            for rule, pred in c['rules_preds']:
                if len(rule) == 0 or (len(rule) == 1 and rule[0]['feature'] == 'default'):
                    if pred == 1:
                        input.loc[valid, 'Prediction'] = current_class
                    continue

                pred_condition = (input['Prediction'] == "N/A")

                for condition in rule:
                    feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

                    if op == '<=':
                        pred_condition &= (input[feat] <= thr)
                    elif op == '>':
                        pred_condition &= (input[feat] > thr)
                    else:
                        print(
                            f"main.py::predict WARNING: Unknown operator! {current_class} {feat} {op} {thr} {len(rule)}")

                if pred == 0:
                    valid &= (~pred_condition)
                else:
                    input.loc[(valid & pred_condition),
                              'Prediction'] = current_class

        return input

    def print_rp_binary(self, rules_preds):
        for rule, pred in rules_preds:
            text = ' AND '.join(
                [f"{condition['feature']} {condition['operator']} {condition['threshold']}" for condition in rule])
            print(f"For rule {text} --> Predicted class: {pred}")

    def print_rules_preds(self):
        if self.multiclass:
            for res in self.result:
                print(f"Rules for class {res['class']}:")
                self.print_rp_binary(res['rules_preds'])
                print()
        else:
            self.print_rp_binary(self.result)