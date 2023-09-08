import pandas as pd
from sequential_covering import SequentialCovering
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import display_data as dd

FOLDER = './report/NO_OS/histology'
MULTICLASS = True
FILE = '/data.csv'
K = 5

data = pd.read_csv(FOLDER + FILE, index_col=False)

PREDICTION_LABEL = data.columns[-1]

# Splitting the data into train and test sets (80% train, 20% test)
train_valid_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

for df in [train_valid_data, test_data]:
    df.reset_index(drop=True, inplace=True)

X_train_valid, y_train_valid = train_valid_data.iloc[:, :-1], train_valid_data.iloc[:, -1]

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
val_accuracies = []
for fold_idx, fold in enumerate(skf.split(X_train_valid, y_train_valid)):
    train_index, val_index = fold
    X_train_fold, X_val_fold = X_train_valid.iloc[train_index], X_train_valid.iloc[val_index]
    y_train_fold, y_val_fold = y_train_valid[train_index], y_train_valid[val_index]
    
    train_data = pd.concat([X_train_fold, pd.DataFrame(y_train_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
    train_data.columns = data.columns

    valid_data = pd.concat([X_val_fold, pd.DataFrame(y_val_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
    valid_data.columns = data.columns

    sc = SequentialCovering(train_data, multiclass=MULTICLASS, max_depth=7, min_samples_leaf=2, output_name=PREDICTION_LABEL)
    sc.fit()

    preds = sc.predict_tmp(valid_data)
    mean_acc = (preds[PREDICTION_LABEL] == preds['Prediction']).mean()
    val_accuracies.append(mean_acc)
    
kf_scores = {'test_score': np.array(val_accuracies)}
dd.visualize_cv(K, kf_scores, FOLDER, f'seqcov_')

# TEST
sc = SequentialCovering(train_valid_data, multiclass=MULTICLASS, max_depth=7, min_samples_leaf=2, output_name=PREDICTION_LABEL)
sc.fit()

print("Learned Rules:")
sc.print_rules_preds()

preds = sc.predict_tmp(test_data)
y_true, y_pred = [tc[0] for tc in pd.DataFrame(preds.iloc[:, -2]).to_numpy()], [pc[0] for pc in pd.DataFrame(preds.iloc[:, -1]).to_numpy()]
dd.visualize_cr_cm(y_true, y_pred, FOLDER, f'seqcov_')