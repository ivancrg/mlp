import numpy as np

# Kernel function that measures the
# similarity of two datapoints
# Returns the value of RBF kernel

def k(xi, xj, gamma):    
    return np.exp(-gamma * np.linalg.norm(xi - xj))

# Computes witness function using list of datapoints,
# prototypes and hyperparameter gamma

def witness(args):
    X, prototypes, x, idx, gamma = args
    w = np.average([k(x, xi, gamma) for xi in X]) - np.average([k(x, zi, gamma) for zi in prototypes])
    return [x, w, idx]