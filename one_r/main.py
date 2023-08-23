import pandas as pd


def calculate_error(predictor, result, value):
    # How often does each result (result) appear for the given value of the predictor
    value_counts = result[predictor == value].value_counts()

    # The most frequent class in relation to this predictor's value
    # (The class (result) that is most commonly seen in combination with this predictor's value)
    most_frequent_class = value_counts.idxmax()

    # Calculate the total error (misclassifications) for this value of the predictor
    error = len(result[predictor == value]) - value_counts[most_frequent_class]

    return value, error, most_frequent_class


def one_r(X, y):
    # Initial parameter setup
    best_predictor = None
    best_error = float('inf')

    # Try all classes as a feature
    for feature in X.columns:
        print(f'Calculating errors for feature {feature}')

        # Find the total error of the selected feature for each possible feature value
        total = [calculate_error(X[feature], y, value)
                 for value in X[feature].unique()]
        print(total)

        # With optimum setup, how much misclassifications will a predictor based on this feature have?
        total_error = sum(e for _, e, _ in total)

        if total_error < best_error:
            best_predictor = {
                'feature': feature,
                'values': [v for v, _, _ in total],
                'result': [r for _, _, r in total]
            }

            best_error = total_error

        print(
            f'{feature} predictor accuracy: {round((len(X) - total_error) / len(X), 3)}')

    return best_predictor


def one_r_multiclass(X, y):
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
        rule = one_r(X, yc)
        rules.append({'class': c, 'rule': rule})

        # Removing the class for which the rule has been calculated
        classes_counts.remove(classes_counts[0])

    # Adding a default rule
    rules.append({'class': classes_counts[0][0], 'rule': 'default'})

    return rules


data = pd.read_csv('../encoded_categorical.csv', index_col=False)
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# best_predictor = one_r(X, y)
# print("Best predictor: ", best_predictor)

rules = one_r_multiclass(X, y)
print("Learned Rules:")
for rule in rules:
    print(f"For class '{rule['class']}': {rule['rule']}")
