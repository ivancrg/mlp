param_grid = {
    'n_estimators': range(25, 101, 25),
    'max_leaf_nodes': range(10, 21, 5),
    'max_features': range(2, 9, 3),
    'max_depth': range(6, 11, 2),
    'min_samples_split': range(10, 21, 10),
    'min_samples_leaf': range(5, 11, 5)
}

print([i for i in param_grid['n_estimators']])
print([i for i in param_grid['max_leaf_nodes']])
print([i for i in param_grid['max_features']])
print([i for i in param_grid['max_depth']])
print([i for i in param_grid['min_samples_split']])
print([i for i in param_grid['min_samples_leaf']])