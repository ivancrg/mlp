import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../encoded_categorical.csv', index_col=False)
print(data.head())
X, y = data.iloc[:, :-1], data.iloc[:, -1]

data_encoded = data.copy()
label_encoder = LabelEncoder()
for colname in data.columns:
    data_encoded[colname] = label_encoder.fit_transform(data[colname])
print(data_encoded.head())

data_decoded = pd.DataFrame()
for colname in data_encoded.columns:
    data_decoded[colname] = label_encoder.inverse_transform(data_encoded[colname])
print(data_decoded.head())



# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
# tree.plot_tree(clf, feature_names=data.columns)
# plt.show()