import pandas as pd
import numpy as np
import mmd
import time

data = pd.read_csv('encoded.csv')

data = data.loc[0:49]
data = data.to_numpy()

# Adds a prototype to list prototypes
# using the datapoints from list X

def add_prototype(X, prototypes, gamma):
    min_mmd = float('+inf')
    
    # Best prototype index (lowers mmd most)
    bpi = 0
    
    # Consider each datapoint as prototype and evaluate like that
    for idx, p in enumerate(X):
        # Creating a copy of datapoints and prototypes
        Xi = X.copy()
        pi = prototypes.copy()
        
        # Removing datapoint from list of datapoints
        Xi = np.delete(Xi, idx, axis=0)
        
        # Making the removed datapoint a prototype
        pi = np.append(pi, p)
        
        # Calculating mmd of new prototype list
        mmd_i = mmd.mmd(Xi, pi, gamma)
        
        # Checking if new mmd is lower
        if mmd_i < min_mmd:
            min_mmd = mmd_i
            bpi = idx
    
    # Making bpith datapoint a prototype in source lists
    prototypes = np.vstack((prototypes, X[bpi]))
    X = np.delete(X, bpi, 0)
    
    return X, prototypes

# Finds m prototypes from dataset X
# Hyperparameters m and gamma

def find_prototypes(X, m=10, gamma=0.1):
    # List of chosen prototypes
    cp = np.empty((0, data.shape[1]))
    
    # Find m prototypes
    for _ in range(m):
        X, cp = add_prototype(X, cp, gamma)
    
    return cp

t = time.time()
prototypes = find_prototypes(data)
print(prototypes)
print(time.time() - t)