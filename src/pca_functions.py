# src/pca_functions.py
"""
Functions to perform PCA, for pca_script.py
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import nibabel as nib # to get the nii format

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

def pca1_normalize(X, select=VarianceThreshold(), scale=StandardScaler()):
    """
    """
    selector = select
    X_filtered = selector.fit_transform(X)
    removed_features_mask = selector.get_support(indices=False)  # Mask of features that were NOT removed
    scaler = scale
    X_scaled = scaler.fit_transform(X_filtered)
    return  X_scaled, scaler, removed_features_mask

def pca1_reconstruct(X_reduced, X, pca, n, scaler, removed_features_mask):
    """
    """
    # Transformed data (only first n columns, shape (30,n)) times eigen_vectors (only first n rows, shape(n,>100000) -> shape (30, >100000), OK!
    X_reconstructed_scaled = X_reduced[:, :n] @ pca.components_[:n, :]
    # Reverse normalization
    X_reconstructed_filtered = scaler.inverse_transform(X_reconstructed_scaled)
    # Re-add constant features removed before PCA
    X_reconstructed = np.zeros_like(X)
    X_reconstructed[:, removed_features_mask] = X_reconstructed_filtered # features not removed
    removed_features_mean = np.mean(X[:, ~removed_features_mask], axis=0) # re-calculation of removed features
    X_reconstructed[:, ~removed_features_mask] = np.tile(removed_features_mean, (X.shape[0], 1)) # 
    return X_reconstructed

def pca1_reformat(X_reconstructed, data_array, nii_obj, patient_name_1, n_pc_toreconstruct, save=True):
    """
    """
    X_reconstructed_3d = X_reconstructed.reshape(data_array.shape[-1], *data_array.shape[:-1]) # Shape of the initial 4D data, but with epochs as first dimension
    X_reconstructed_4d = np.transpose(X_reconstructed_3d, (1, 2, 3, 0)) # put epochs as last dimension
    nii_reconstructed = nib.Nifti1Image(X_reconstructed_4d, nii_obj.affine, nii_obj.header)
    if save:
        nib.save(nii_reconstructed, path_resultsfolder+ patient_name_1 + "_projected_" + repr(n_pc_toreconstruct) + "_4d.nii.gz")
    return nii_reconstructed

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
    n_components = len(pca.explained_variance_ratio_)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(patient_name + ': variance explained by principal components') # Title
    # Subplot top: explained variance
    ax1.plot(pca.explained_variance_ratio_, marker='o', linestyle='--')
    # ax1.set_xlabel('Number of principal components')
    ax1.set_ylabel('Explained variance')
    ax1.set_ylim(0, max(pca.explained_variance_ratio_)*1.1)
    ax1.set_xticks(range(n_components))
    ax1.set_xticklabels(range(1, n_components + 1))
    ax1.grid(True)
    # plot bot: cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(cumulative_variance, marker='o', linestyle='--')
    ax2.set_xticks(range(n_components))
    ax2.set_xticklabels(range(1, n_components + 1))
    ax2.set_xlabel('Number of principal components')
    ax2.set_ylabel('Cumulative explained variance')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    #save fig
    plt.savefig(path_resultsfolder + patient_name + "_PCA1_explainedvariance.png")


def plot_pcvalues_2d(X_reduced, pc_n1, pc_n2, patient_str, details_str, axisscale_fixed=True):
    """
    """
    fig, ax1 = plt.subplots(1,1) 
    colors = plt.cm.coolwarm(np.linspace(0, 1, X_reduced.shape[0]))
    
     # Scatter plot with progressive colors
    scatter1 = ax1.scatter(
        X_reduced[:, pc_n1],
        X_reduced[:, pc_n2],
        s=40,
        c=np.linspace(0, 1, X_reduced.shape[0]),
        cmap='coolwarm'
    )

    # Plot segments with colormap
    for i in range(X_reduced.shape[0] - 1):
        ax1.plot(
            X_reduced[i:i+2, pc_n1],
            X_reduced[i:i+2, pc_n2],
            color=colors[i],
            linestyle='-',
            linewidth=1
        )

    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label('Time (Epoch)')
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{i}' for i in np.linspace(0, X_reduced.shape[0]-1, 6, dtype=int)])

    ax1.set(
        title=f"{patient_str} : Principal Components {pc_n1+1} and {pc_n2+1}",
         xlabel=f"Principal Component {pc_n1+1}",
        ylabel=f" Principal Component {pc_n2+1}")

    if axisscale_fixed :
        axis1, axis2 = 0, 1 
    else:
         axis1, axis2 = pc_n1, pc_n2      
    x_ticks = np.linspace(-max(abs(min(X_reduced[:, axis1])), max(X_reduced[:, axis1])), max(abs(min(X_reduced[:, axis1])), max(X_reduced[:, axis1])),7)
    y_ticks = np.linspace(-max(abs(min(X_reduced[:, axis2])), max(X_reduced[:, axis2])), max(abs(min(X_reduced[:, axis2])), max(X_reduced[:, axis2])),7)
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    # ax1.xaxis.set_ticklabels([])
    # ax1.yaxis.set_ticklabels([])

    plt.savefig(path_resultsfolder + patient_str + details_str + "_" + repr(pc_n1+1) + "and" + repr(pc_n2+1) + ".png")
