import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import display_data as dd


plt.rcParams.update({'font.size': 14})

folder = './report/NO_OS/histology'
file = '/data.csv'

data = pd.read_csv(folder + file)

train_valid_data, test_data = train_test_split(
    data, test_size=0.2, random_state=47)

X_train_valid, y_train_valid = train_valid_data.iloc[:,
                                                     :-1], train_valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

k = 5
scores = cross_validate(rf_classifier,  X_train_valid,
                        y_train_valid, cv=k, return_estimator=True)

dd.visualize_cv(k, scores, folder, 'rf_')


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_valid, y_train_valid)
y_test_pred = rf_classifier.predict(X_test)
dd.visualize_cr_cm(y_test, y_test_pred, folder, 'rf_')

plt.show()
