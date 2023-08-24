import pandas as pd
from sequential_covering import SequentialCovering
from sklearn.model_selection import train_test_split

data = pd.read_csv('../encoded.csv', index_col=False)

# Splitting the data into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

for df in [train_data, test_data]:
    df.reset_index(drop=True, inplace=True)

sc = SequentialCovering(train_data, True)
sc.fit()

print("Learned Rules:")
sc.print_rules_preds()

preds = sc.predict(test_data.copy())
print(preds.head())
print((preds['Preoperative Diagnosis'] == preds['Prediction']).mean())