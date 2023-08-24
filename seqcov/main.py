import pandas as pd
from sequential_covering import SequentialCovering

data = pd.read_csv('../encoded_binary.csv', index_col=False)
data.reset_index(drop=True, inplace=True)

sc = SequentialCovering(data, False)
sc.fit()

print("Learned Rules:")
sc.print_rules_preds()

print(sc.predict(data.head().copy()))