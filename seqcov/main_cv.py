import pandas as pd
from sequential_covering import SequentialCovering
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
import display_data as dd

folder = './report/NO_OS/postop'
file = '/data.csv'
k = 5

data = pd.read_csv(folder + file, index_col=False)

# Splitting the data into train and test sets (80% train, 20% test)
train_valid_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

for df in [train_valid_data, test_data]:
    df.reset_index(drop=True, inplace=True)

X_train_valid, y_train_valid = train_valid_data.iloc[:, :-1], train_valid_data.iloc[:, -1]

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
val_accuracies = []
for fold_idx, fold in enumerate(skf.split(X_train_valid, y_train_valid)):
    train_index, val_index = fold
    X_train_fold, X_val_fold = X_train_valid.iloc[train_index], X_train_valid.iloc[val_index]
    y_train_fold, y_val_fold = y_train_valid[train_index], y_train_valid[val_index]
    
    train_data = pd.concat([X_train_fold, pd.DataFrame(y_train_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
    train_data.columns = data.columns

    valid_data = pd.concat([X_val_fold, pd.DataFrame(y_val_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
    valid_data.columns = data.columns

    sc = SequentialCovering(train_data, True, max_depth=7, min_samples_leaf=2, output_name='Postoperative Diagnosis')
    sc.fit()

    print("Learned Rules:")
    sc.print_rules_preds()

    preds = sc.predict(valid_data)
    mean_acc = (preds['Postoperative Diagnosis'] == preds['Prediction']).mean()
    val_accuracies.append(mean_acc)
    print(f"Fold {fold_idx + 1} accuracy: {mean_acc}")
    y_true, y_pred = [tc[0] for tc in pd.DataFrame(preds.iloc[:, -2]).to_numpy()], [pc[0] for pc in pd.DataFrame(preds.iloc[:, -1]).to_numpy()]
    
kf_scores = {'test_score': np.array(val_accuracies)}
dd.visualize_cv(k, kf_scores, folder, f'seqcov_')

# TEST

# preds = sc.predict(valid_data)
# print((preds['Postoperative Diagnosis'] == preds['Prediction']).mean())
# print(preds['Prediction'].value_counts())
# y_true, y_pred = [tc[0] for tc in pd.DataFrame(preds.iloc[:, -2]).to_numpy()], [pc[0] for pc in pd.DataFrame(preds.iloc[:, -1]).to_numpy()]

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test_class, y_pred_class))

# from sklearn.metrics import classification_report
# print(classification_report(y_test_class, y_pred_class))