import os
import log
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mmd
import witness
import joblib
import time
import matplotlib
matplotlib.use('Agg')

LOG_DIR = './log_test_spl_cplx_othr'
data_pd = pd.read_csv('test_spl_cplx_othr.csv')


def evaluate_as_prototype(args):
    Xi, pi, idx, gamma = args

    # Making the removed datapoint a prototype
    pi = np.append(pi, Xi[idx])

    # Removing datapoint from list of datapoints
    Xi = np.delete(Xi, idx, axis=0)

    # Calculating mmd of new prototype list
    # Returning pair (mmd, idx) of the
    # evaluated datapoint as prototype
    return np.array([mmd.mmd(Xi, pi, gamma), idx])


def select_prototype(datapoints):
    return joblib.Parallel(n_jobs=16)(joblib.delayed(evaluate_as_prototype)(datapoint) for datapoint in datapoints)


# Adds a prototype to list prototypes
# using the datapoints from list X

def add_prototype(X, prototypes, gamma):
    min_mmd = float('+inf')

    bpi = 0

    datapoints = [(X.copy(), prototypes.copy(), idx, gamma)
                  for idx, _ in enumerate(X)]

    # Consider each datapoint as prototype and evaluate like that
    mmds = np.array(select_prototype(datapoints))

    # Best prototype index (lowers mmd most)
    bpi = int(mmds[np.argmin(mmds[:, 0]), 1])

    # Making bpith datapoint a prototype in source lists
    prototypes = np.vstack((prototypes, X[bpi]))
    X = np.delete(X, bpi, 0)

    return X, prototypes


# Finds m prototypes from dataset X
# Hyperparameters m and gamma

def find_prototypes(X, m, gamma):
    # List of chosen prototypes
    cp = np.empty((0, data.shape[1]))

    # Find m prototypes
    for _ in range(m):
        X, cp = add_prototype(X, cp, gamma)

    return X, cp


# Finds m criticisms from dataset X with given prototypes
# Hyperparameters m and gamma

def find_criticisms_par(X, prototypes, m, gamma):
    # List of witness values
    wv = joblib.Parallel(n_jobs=12)(joblib.delayed(witness.witness)(
        [np.delete(X, idx, axis=0), prototypes, x, idx, gamma]) for idx, x in enumerate(X))

    return sorted(wv, key=lambda x: -abs(x[1]))[0:m]


def visualize(data, prototypes, criticisms, fig2dloc=None, fig3dloc=None):
    # Finding indices of prototypes and criticisms to mark them in plots
    prototype_indices = np.argwhere(
        np.isin(data, prototypes).all(axis=1)).flatten()
    criticism_indices = np.argwhere(
        np.isin(data, [c[0] for c in criticisms]).all(axis=1)).flatten()

    # Perform t-SNE on the dataset for 2D visualization
    tsne_2d = TSNE(n_components=2, random_state=42)
    data_2d = tsne_2d.fit_transform(data)

    # Plot 2D visualization
    plt.figure()
    plt.scatter(data_2d[:, 0], data_2d[:, 1])

    # Mark prototypes as red triangles
    plt.scatter(data_2d[prototype_indices, 0], data_2d[prototype_indices,
                1], color='red', marker='^', s=100, label='Prototypes')

    # Mark criticisms as blue circles
    plt.scatter(data_2d[criticism_indices, 0], data_2d[criticism_indices,
                1], color='blue', marker='o', s=100, label='Criticisms')

    plt.title('t-SNE Visualization (2D)')
    plt.legend()

    # Save the figure to a variable
    if fig2dloc is not None:
        plt.savefig(fig2dloc, format='png')

    plt.close()

    # Perform t-SNE on the dataset for 3D visualization
    tsne_3d = TSNE(n_components=3, random_state=42)
    data_3d = tsne_3d.fit_transform(data)

    # Plot 3D visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2])

    # Mark prototypes as red triangles
    ax.scatter(data_3d[prototype_indices, 0], data_3d[prototype_indices, 1],
               data_3d[prototype_indices, 2], color='red', marker='^', s=100, label='Prototypes')

    # Mark criticisms as blue circles
    ax.scatter(data_3d[criticism_indices, 0], data_3d[criticism_indices, 1],
               data_3d[criticism_indices, 2], color='blue', marker='o', s=100, label='Criticisms')

    ax.set_title('t-SNE Visualization (3D)')
    plt.legend()

    # Save the figure to a variable
    if fig3dloc is not None:
        plt.savefig(fig3dloc, format='png')

    plt.close()


# Selecting number of instances
a = ''
for n in np.linspace(30, len(data_pd), 3):
    n = int(n)
    data = data_pd.loc[0:n]
    data = data.to_numpy()

    # Selecting number of prototypes
    for m_proto in np.linspace(3, 20, 3):
        m_proto = int(m_proto)

        # Selecting number of criticisms
        for m_crit in np.linspace(3, 20, 3):
            m_crit = int(m_crit)

            # Selecting gamma
            for gamma in range(5, 106, 20):
                gamma /= 100

                folder_name = f'n={len(data)}_mproto={m_proto}_mcrit={m_crit}_gamma={gamma}'
                loc = os.path.join(LOG_DIR, folder_name)
                os.mkdir(loc)

                t_proto = time.time()
                X, prototypes = find_prototypes(data, m=m_proto, gamma=gamma)
                t_proto = time.time() - t_proto

                t_criti = time.time()
                criticisms = find_criticisms_par(
                    X, prototypes, m=m_crit, gamma=gamma)
                t_criti = time.time() - t_criti

                visualize(data, prototypes, criticisms, os.path.join(
                    loc, 'plot2d.png'), os.path.join(loc, 'plot3d.png'))

                log.save_search(loc, data, prototypes, t_proto,
                                criticisms, t_criti, m_proto, m_crit, gamma)
