import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../encoded.csv')

train_data, remaining_data = train_test_split(data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(remaining_data, test_size=0.35, random_state=42)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

dt_classifier = DecisionTreeClassifier(random_state=42)

k  = 5
scores = cross_validate(dt_classifier, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), cv=k, return_estimator=True)

for fold, score in enumerate(scores['test_score']):
    print(f"Fold {fold+1} accuracy: {score}")

print(f"Average accuracy: {scores['test_score'].mean()}")

plt.figure(1, figsize=(10, 6))
bars = plt.bar(range(1, k + 1), scores['test_score'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 5), ha='center', va='bottom')

plt.xlabel('Fold')
plt.ylabel('Accuracy')

for fold, t in enumerate(scores['estimator']):
    print(f"Plotting fold {fold+1} tree")
    plt.figure(f'Fold {fold + 1}\'s tree')
    tree.plot_tree(t, filled=True)

plt.show()