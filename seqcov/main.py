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
    for col_name, value in rule:
        remove_condition &= (data_encoded[col_name] == value)
    
    # Filtering instances which should be removed (because they're covered by the new_rule)
    new_data = data[~remove_condition]

    return new_data

def sc(data):
    rules_preds = []
    
    while len(data) > 0:
        data.reset_index(drop=True, inplace=True)
        
        lor = LearnOneRule(data.copy(), max_depth=5, min_samples_leaf=1)
        _, new_rule, pred, n_covered = lor.learn_one_rule(0, None, None, [])
        plt.show()
        
        # Append a rule as a tuple of an array of (feature, value) conditions and outcome
        rules_preds.append((new_rule, pred))

        # Removing covered instances
        data = remove_covered_instances(data, new_rule)

    return rules_preds

def sc_multiclass(data):
    # Array of tuples - (class, number of instances of that class)
    y = pd.DataFrame(data.iloc[:, -1])
    classes_counts = [(c, sum(y.iloc[:, 0] == c)) for c in y.iloc[:, 0].unique()]
    classes_counts.sort(key=lambda x: x[1])

    rules = []

    print(classes_counts, y.value_counts(ascending=True))

    # While there are classes for which the rule hasn't been set yet
    while len(classes_counts) > 1:
        data.reset_index(drop=True, inplace=True)

        # Current class for which a rule should be set
        current_class = classes_counts[0][0]

        print(f'Current class: {current_class}')

        # Copy of the dataset to be modified to binary classification
        # yc = y.copy()
        # yc.loc[y.iloc[:, 0] == current_class] = 1
        # yc.loc[y.iloc[:, 0] != current_class] = 0
        data_current = data.copy()
        data_current.loc[data.iloc[:, -1] == current_class, 'Preoperative Diagnosis'] = 1
        data_current.loc[data.iloc[:, -1] != current_class, 'Preoperative Diagnosis'] = 0
        print(data_current.loc[data_current['Preoperative Diagnosis'] == 1])

        # if current_class == 3:
        #     for i in range(len(data_current)):
        #         if data_current.iloc[i, -1] != 0:
        #             print(data_current.iloc[i])

        # Calculating the rule for the current class
        rules_preds_bin = sc(data_current)
        # rules_binary = sc(X.merge(yc, left_index=True, right_index=True))

        # Remove all covered instances using rules_binary
        for r, pred in rules_preds_bin:
            # We remove only positive data instances
            if pred == 1:
                data = remove_covered_instances(data, r)

        # Add the new rules to result
        rules.append({'class': current_class, 'rules': rules_preds_bin})

        # Removing the class for which the rule has been calculated
        classes_counts.remove(classes_counts[0])

    # Adding a default rule
    rules.append({'class': classes_counts[0][0], 'rules': [('default', classes_counts[0][0])]})

    return rules

# data = pd.read_csv('../encoded_categorical_binary.csv', index_col=False)
data = pd.read_csv('../encoded_categorical.csv', index_col=False)
data.reset_index(drop=True, inplace=True)

# rules_preds = sc(data)

# print("Learned Rules:")
# for rule, pred in rules_preds:
#     print(f"For rule {rule} --> Predicted class: {pred}")

result = sc_multiclass(data)

print("Learned Rules:")
for res in result:
    print(f"Rules for class {res['class']}:")
    print(res['rules'])
    for rule, pred in res['rules']:
        print(f"For rule {rule} --> Predicted class: {pred}")