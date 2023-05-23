import numpy as np
from itertools import combinations
import multiprocessing
import joblib

# Kernel function that measures the
# similarity of two datapoints
# Returns the value of RBF kernel

def k(xi, xj, gamma):    
    return np.exp(-gamma * np.linalg.norm(xi - xj))

def k_par(args):
    xi, xj, gamma = args    
    return np.exp(-gamma * np.linalg.norm(xi - xj))

# Computes mmd^2 using list of datapoints,
# prototypes and hyperparameter gamma

def mmd(X, prototypes, gamma):
    # Sum of protoype proximities
    # Generate combinations of row indices
    cmb = list(combinations(np.arange(prototypes.shape[0]), 2))
    
    # Extract rows based on combinations
    # Calculate sum of proximities of combinations
    # Needs to be multiplied with 2 (lower part of proximity matrix)
    # Sum of identity matrix needs to be added (1 on diag)
    utm = np.sum(list(map(lambda z: k(z[0], z[1], gamma), prototypes[cmb])))
    spp = utm * 2 + prototypes.shape[0]
    
    
    # Sum of prototype-datapoint proximities
    spdp = np.sum([[k(z, x, gamma) for z in prototypes] for x in X])
    #for i in prototypes.index:
    #    for j in X.index:
    #        spdp += k(prototypes.loc[i], X.loc[j], gamma)
    
    
    # Sum of datapoint proximities
    # Generate combinations of row indices
    cmb = list(combinations(np.arange(X.shape[0]), 2))
    
    # Extract rows based on combinations
    # Calculate sum of proximities of combinations
    # Needs to be multiplied with 2 (lower part of proximity matrix)
    # Sum of identity matrix needs to be added (1 on diag)
    utm = np.sum(list(map(lambda x: k(x[0], x[1], gamma), X[cmb])))
    sdp = utm * 2 + X.shape[0]
    
    
    # Averaging proximities to compute mmd^2
    mmd2 = 1/(prototypes.shape[0]**2) * spp - 2/(prototypes.shape[0]*X.shape[0]) * spdp + 1/(X.shape[0]**2) * sdp
    
    return mmd2

def mmd_parallel(X, prototypes, gamma):
    # Sum of protoype proximities
    # Generate combinations of row indices
    cmb = list(combinations(np.arange(prototypes.shape[0]), 2))
    
    # Extract rows based on combinations
    # Calculate sum of proximities of combinations
    # Needs to be multiplied with 2 (lower part of proximity matrix)
    # Sum of identity matrix needs to be added (1 on diag)
    pool_obj = multiprocessing.Pool(4)
    ans = pool_obj.map(k_par, [[p[0], p[1], gamma] for p in prototypes[cmb]])
    utm = np.sum(ans)
    spp = utm * 2 + prototypes.shape[0]
    pool_obj.close()
    
    
    # Sum of prototype-datapoint proximities
    spdp = np.sum([[k(z, x, gamma) for z in prototypes] for x in X])
    
    
    # Sum of datapoint proximities
    # Generate combinations of row indices
    cmb = list(combinations(np.arange(X.shape[0]), 2))
    
    # Extract rows based on combinations
    # Calculate sum of proximities of combinations
    # Needs to be multiplied with 2 (lower part of proximity matrix)
    # Sum of identity matrix needs to be added (1 on diag)
    pool_obj = multiprocessing.Pool(4)
    ans = pool_obj.map(k_par, [[x[0], x[1], gamma] for x in X[cmb]])
    utm = np.sum(ans)
    sdp = utm * 2 + X.shape[0]
    pool_obj.close()
    
    
    # Averaging proximities to compute mmd^2
    mmd2 = 1/(prototypes.shape[0]**2) * spp - 2/(prototypes.shape[0]*X.shape[0]) * spdp + 1/(X.shape[0]**2) * sdp
    
    return mmd2

def mmd_joblib(X, prototypes, gamma):
    # Sum of protoype proximities
    # Generate combinations of row indices
    cmb = list(combinations(np.arange(prototypes.shape[0]), 2))
    
    # Extract rows based on combinations
    # Calculate sum of proximities of combinations
    # Needs to be multiplied with 2 (lower part of proximity matrix)
    # Sum of identity matrix needs to be added (1 on diag)
    ans = joblib.Parallel(n_jobs=12)(joblib.delayed(k)(p[0], p[1], gamma) for p in prototypes[cmb])
    utm = np.sum(ans)
    spp = utm * 2 + prototypes.shape[0]
    
    
    # Sum of prototype-datapoint proximities
    spdp = np.sum([[k(z, x, gamma) for z in prototypes] for x in X])
    
    
    # Sum of datapoint proximities
    # Generate combinations of row indices
    cmb = list(combinations(np.arange(X.shape[0]), 2))
    
    # Extract rows based on combinations
    # Calculate sum of proximities of combinations
    # Needs to be multiplied with 2 (lower part of proximity matrix)
    # Sum of identity matrix needs to be added (1 on diag)
    ans = joblib.Parallel(n_jobs=12)(joblib.delayed(k)(x[0], x[1], gamma) for x in X[cmb])
    utm = np.sum(ans)
    sdp = utm * 2 + X.shape[0]
    
    
    # Averaging proximities to compute mmd^2
    mmd2 = 1/(prototypes.shape[0]**2) * spp - 2/(prototypes.shape[0]*X.shape[0]) * spdp + 1/(X.shape[0]**2) * sdp
    
    return mmd2