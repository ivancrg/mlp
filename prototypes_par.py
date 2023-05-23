import pandas as pd
import numpy as np
import mmd
import concurrent.futures
import time

data = pd.read_csv('encoded.csv')

data = data.loc[0:49]
data = data.to_numpy()

def evaluate_as_prototype(args):
    Xi, pi, idx, gamma = args
    #Xi = X.copy()
    #pi = p.copy()

    # Making the removed datapoint a prototype
    pi = np.append(pi, Xi[idx])

    # Removing datapoint from list of datapoints
    Xi = np.delete(Xi, idx, axis=0)

    # Calculating mmd of new prototype list
    # Returning pair (mmd, idx) of the
    # evaluated datapoint as prototype
    return np.array([mmd.mmd(Xi, pi, gamma), idx])

def select_prototype(datapoints):
    # Number of threads to use
    num_threads = 12

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit the tasks to the executor
        futures = [executor.submit(evaluate_as_prototype, item) for item in datapoints]

        # Wait for the tasks to complete
        concurrent.futures.wait(futures)
        
        return [future.result() for future in futures]

# Adds a prototype to list prototypes
# using the datapoints from list X

def add_prototype(X, prototypes, gamma):
    min_mmd = float('+inf')
    
    bpi = 0
    
    datapoints = [(X.copy(), prototypes.copy(), idx, gamma) for idx, _ in enumerate(X)]
    
    # Consider each datapoint as prototype and evaluate like that
    mmds = np.array(select_prototype(datapoints))
    
    # Best prototype index (lowers mmd most)
    bpi = int(mmds[np.argmin(mmds[:, 0]), 1])
    # for mmd, idx in mmds:
    #     # Checking if new mmd is lower
    #     if mmd < min_mmd:
    #         min_mmd = mmd
    #         bpi = idx
    
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