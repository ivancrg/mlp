import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.inspection import permutation_importance
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import display_data as dd 

plt.rcParams.update({'font.size': 14})

folder = './report/NO_OS/histology_binary'
file = '/data.csv'

data = pd.read_csv(folder + file)

train_valid_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42)

X_train_valid, y_train_valid = train_valid_data.iloc[:,
                                                     :-1], train_valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

dt_classifier = DecisionTreeClassifier(random_state=42)

k = 5
scores = cross_validate(dt_classifier, X_train_valid,
                        y_train_valid, cv=k, return_estimator=True)

dd.visualize_cv(k, scores, folder, prefix='dt_')

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_valid, y_train_valid)

y_test_pred = dt_classifier.predict(X_test)
dd.visualize_cr_cm(y_test, y_test_pred, folder, prefix='dt_')

plt.show()

plt.figure(figsize=(12, 10))

results = permutation_importance(
    dt_classifier,
    X_train_valid,
    y_train_valid,
    n_repeats=10,
    random_state=42
    )

importance = results.importances_mean

for i, v in enumerate(importance):
    print(f'Feature {i}: {v:.5f}')

plt.subplots_adjust(left=0.09, right=0.96, bottom=0.33, top=0.97)

plt.bar(X_test.columns, importance)
plt.xticks(rotation=90)
plt.ylabel('Importance')
plt.savefig(folder + '/dt_fimp.png')