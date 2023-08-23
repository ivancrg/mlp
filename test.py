import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../encoded.csv', index_col=False).head()

print(data)


# CSV row 2
# print(clf.predict([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]]))

# CSV row 417
# print(clf.predict([[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]]))
