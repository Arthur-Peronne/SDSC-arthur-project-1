# src/pca_functions.py
"""
Functions to perform PCA, for pca_script.py
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from paths import *

# PCA 1: each 3D image as a sample (how voxels co-vary over time, temporal dynamics) -> 30 lines, >100 000 columns (dimensions).
def pca1_transpose(data_array, print_infos=True):
    """
    From 4D numpy array to a 2D aarray (30, >100000)
    """
    data_transposed = np.transpose(data_array, (3, 0, 1, 2))
    X = data_transposed.reshape(30, -1)
    if print_infos:
        print("Shape of X:", X.shape)  # Should be (30, >100000)
    return X

def pca_clean(X):
    """
    Use StandardScaler not to introduce NaNs with divisions with variances = 0
    """
    # Clean data to remove constant features
    selector = VarianceThreshold() 
    X_filtered = selector.fit_transform(X)
     # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    return X_scaled

def plot_pca_explipower(pca,patient_name):
    """
    """
    # fig
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(patient_name + ': variance explained by principal components') # Title
    # Subplot top: explained variance
    ax1.plot(pca.explained_variance_ratio_, marker='o', linestyle='--')
    # ax1.set_xlabel('Number of principal components')
    ax1.set_ylabel('Explained variance')
    ax1.set_ylim(0, max(pca.explained_variance_ratio_)*1.1)
    ax1.grid(True)
    # plot bot: cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(cumulative_variance, marker='o', linestyle='--')
    ax2.set_xlabel('Number of principal components')
    ax2.set_ylabel('Cumulative explained variance')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    #save fig
    plt.savefig(path_resultimagesfolder + patient_name + "_PCA1_explainedvariance.png")