import pandas as pd
from sklearn import tree

def learn_one_rule(X, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

def sc(data):
    rules = []

    while len(data) > 0:
        # Append a rule as a tuple of an array of (feature, value) conditions and outcome
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        new_rule = learn_one_rule(X, y)
        rules.append(new_rule)

        # Filtering instances which should be removed (because they're covered by the new_rule)
        data_to_remove = data.copy()
        for condition in new_rule['conditions']:
            feature, value = condition
            print(f'Filtering for {feature} = {value}')
            data_to_remove = data_to_remove.loc[feature == value]
        
        # Removing covered instances from data
        data = data[~data.isin(data_to_remove.to_dict('list')).all(axis=1)]

    return rules

def sc_multiclass(X, y):
    # Array of tuples - (class, number of instances of that class)
    classes_counts = [(c, sum(y == c)) for c in y.unique()]
    classes_counts.sort(key=lambda x: x[1])

    rules = []

    print(classes_counts, y.value_counts(ascending=True))

    # While there are classes for which the rule hasn't been set yet
    while len(classes_counts) > 1:
        # Current class for which a rule should be set
        c = classes_counts[0][0]

        # Copy of the dataset to be modified to binary classification
        yc = y.copy()
        yc.loc[yc == c] = 1
        yc.loc[yc != c] = 0

        # Calculating the rule for the current class
        rules_binary = sc(X, yc)
        rules.append({'class': c, 'rules': rules_binary})

        # Removing the class for which the rule has been calculated
        classes_counts.remove(classes_counts[0])

    # Adding a default rule
    rules.append({'class': classes_counts[0][0], 'rule': 'default'})

    return rules

data = pd.read_csv('../encoded_categorical.csv', index_col=False)
X, y = data.iloc[:, :-1], data.iloc[:, -1]

rules = sc_multiclass(X, y)
# print("Learned Rules:")
# for rule in rules:
#     print(f"For class '{rule['class']}': {rule['rule']}")