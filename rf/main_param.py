import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('./final/postop_spl_cplx_othr.csv')

train_data, remaining_data = train_test_split(
    data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(
    remaining_data, test_size=0.35, random_state=42)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
metrics = ['score']

k = 5
scores = cross_validate(rf_classifier, pd.concat(
    [X_train, X_valid]), pd.concat([y_train, y_valid]), cv=k)

for metric_idx, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.bar(range(k), scores[f'test_{metric}'])
    plt.xlabel('Fold')
    plt.ylabel(metric)
    plt.title(f'test_{metric} for each fold')
    plt.show()
