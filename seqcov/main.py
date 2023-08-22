import pandas as pd
from sklearn import tree
from learn_one_rule import LearnOneRule
import matplotlib.pyplot as plt


def learn_one_rule(X, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

# rule - list of (column, value) tuples used to filter the data


def remove_covered_instances(data, rule):
    data.reset_index(drop=True, inplace=True)

    # print(f'Data length: {len(data)}')

    # One-hot encoding (rules are based on OHE-d data)
    data_encoded = pd.get_dummies(data)

    # Creating condition for removing instances
    # (all instances under remove condition should be removed)
    remove_condition = pd.Series([True] * len(data))
    for condition in rule:
        feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

        if op == '<=':
            remove_condition &= (data_encoded[feat] <= thr)
        elif op == '>':
            remove_condition &= (data_encoded[feat] > thr)
        else:
            print("main.py::remove_covered_instances WARNING: Unknown operator!")

    # Filtering instances which should be removed (because they're covered by the new_rule)
    new_data = data[~remove_condition]

    return new_data


def sc(data):
    rules_preds = []

    while len(data) > 0:
        data.reset_index(drop=True, inplace=True)

        lor = LearnOneRule(data.copy(), max_depth=3, min_samples_leaf=1)
        _, new_rule, pred, n_covered = lor.learn_one_rule(
            0, None, None, None, [])
        # lor.plot_classifier()
        # plt.show()

        # Append a rule as a tuple of an array of (feature, value) conditions and outcome
        rules_preds.append((new_rule, pred))

        # Removing covered instances
        data = remove_covered_instances(data, new_rule)

    return rules_preds


def sc_multiclass(data):
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
        rules_preds_bin = sc(data_current)

        # Remove all covered instances using rules_binary
        for r, pred in rules_preds_bin:
            # We remove only positive data instances
            if pred == 1:
                data = remove_covered_instances(data, r)

        # Add the new rules to result
        rules.append({'class': current_class, 'rules_preds': rules_preds_bin})

        # Removing the class for which the rule has been calculated
        classes_counts.remove(classes_counts[0])

    # Adding a default rule
    rules.append({
        'class': classes_counts[0][0],
        'rules_preds': [([{
            'feature': 'default',
            'operator': '',
            'threshold': ''
        }], classes_counts[0][0])]})

    return rules


def print_rules_preds(rules_preds):
    for rule, pred in rules_preds:
        text = ' AND '.join(
            [f"{condition['feature']} {condition['operator']} {condition['threshold']}" for condition in rule])
        print(f"For rule {text} --> Predicted class: {pred}")


# data = pd.read_csv('../encoded_binary.csv', index_col=False)
data = pd.read_csv('../encoded.csv', index_col=False)
data.reset_index(drop=True, inplace=True)

# rules_preds = sc(data)
# print("Learned Rules:")
# print_rules_preds(rules_preds)

result = sc_multiclass(data)
print("Learned Rules:")
for res in result:
    print(f"Rules for class {res['class']}:")
    print_rules_preds(res['rules_preds'])
    print()
