import pandas as pd
from collections import Counter

def calculate_error(predictions, targets):
    return sum(p != t for p, t in zip(predictions, targets))

def one_r_algorithm(dataset, target_column):
    best_rule = None
    best_error = float('inf')

    for feature in dataset.columns:
        if feature == target_column:
            continue

        rule = {}
        counts = dataset.groupby([feature, target_column]).size().unstack(fill_value=0)
        total_per_feature = counts.sum(axis=1)

        for value, sub_counts in counts.iterrows():
            predicted_class = sub_counts.idxmax()
            errors = total_per_feature[value] - sub_counts[predicted_class]
            rule[value] = (predicted_class, errors)

        total_error = sum(error for _, (_, error) in rule.items())
        if total_error < best_error:
            best_rule = rule
            best_error = total_error

    return best_rule

# Example usage:
# Assuming you have a DataFrame named 'data' containing the dataset and 'target_column' as the name of the target class column
data = pd.DataFrame({
    'Feature_1': [1, 2, 3, 1, 2, 3],
    'Feature_2': [0, 1, 0, 1, 0, 1],
    'Target_Class': ['A', 'A', 'B', 'A', 'B', 'B']
})

best_rule = one_r_algorithm(data, 'Target_Class')
print(best_rule)
