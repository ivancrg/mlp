import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('./final/postop_spl_cplx_othr.csv')

train_valid_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42)
train_data, valid_data = train_test_split(
    train_valid_data, test_size=0.2, random_state=42)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

k = 5
scores = cross_val_score(rf_classifier, pd.concat(
    [X_train, X_valid]), pd.concat([y_train, y_valid]), cv=k)

for fold, accuracy in enumerate(scores):
    print(f"Fold {fold+1} accuracy: {accuracy}")

plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, k + 1), scores)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval,
             round(yval, 5), ha='center', va='bottom')


plt.xlabel('Fold')
plt.ylabel('Accuracy')

plt.show()

print(f"Average accuracy: {np.mean(scores)}")
