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
def pca1_transpose(data_array, lines =30, print_infos=True):
    """
    From 4D numpy array to a 2D aarray (30, >100000)
    """
    data_transposed = np.transpose(data_array, (3, 0, 1, 2))
    X = data_transposed.reshape(lines, -1)
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
    ax1.set_xticks(range(0, n_components, int(n_components/30)))
    ax1.set_xticklabels(range(1, n_components + 1, int(n_components/30)))
    ax1.grid(True)
    # plot bot: cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(cumulative_variance, marker='o', linestyle='--')
    ax2.set_xticks(range(0, n_components, int(n_components/30)))
    ax2.set_xticklabels(range(1, n_components + 1, int(n_components/30)))
    ax2.set_xlabel('Number of principal components')
    ax2.set_ylabel('Cumulative explained variance')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    #save fig
    plt.savefig(path_resultsfolder + patient_name + "_PCA_explainedvariance.png")


def plot_pcvalues_2d(X_reduced, pc_n1, pc_n2, patient_str, details_str, scale_str = 'Time (Epoch)', segments = True, axisscale_fixed=True):
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
    if segments:
        for i in range(X_reduced.shape[0] - 1):
            ax1.plot(
                X_reduced[i:i+2, pc_n1],
                X_reduced[i:i+2, pc_n2],
                color=colors[i],
                linestyle='-',
                linewidth=1
            )

    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label(scale_str)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{i+1}' for i in np.linspace(0, X_reduced.shape[0]-1, 6, dtype=int)])

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


# PCA2 : spatial 

def pca2_reformat(X_reconstructed, data_array, nii_obj_template, patient_index):
    """
    """
    # Reconstruct 4D (all patients)
    n_patients = data_array.shape[0]
    spatial_shape = data_array.shape[1:]  # (256,256,10)
    img4d = X_reconstructed.reshape(n_patients, *spatial_shape)  # (n,256,256,10)
    # Get image 3D of the patient chosen
    img3d = img4d[patient_index]  # (256,256,10)
    nii = nib.Nifti1Image(img3d, nii_obj_template.affine, nii_obj_template.header)
    return nii



def plot_pcvalues_2d_meta(X_reduced, pc_n1, pc_n2, metainfo_str, metainfo_list, axisscale_fixed=True, extremes_toremove=15):
    """
    """
    fig, ax1 = plt.subplots(1,1) 

    # Color scale for numeric
    botlimit, toplimit = sorted(metainfo_list)[extremes_toremove],  sorted(metainfo_list)[-extremes_toremove]
    print(botlimit, toplimit)
    colors = [max(min((meta_info - botlimit)/(toplimit - botlimit),1),0) for meta_info in metainfo_list]

     # Scatter plot with progressive colors
    scatter1 = ax1.scatter(
        X_reduced[:, pc_n1],
        X_reduced[:, pc_n2],
        s=40,
        c= colors,
        cmap='coolwarm'
    )

    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label(metainfo_str)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{i}' for i in np.linspace(botlimit, toplimit, 6)])

    ax1.set(
        title=f"Principal Components {pc_n1+1} and {pc_n2+1} and correlation with patient {metainfo_str}",
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

    plt.savefig(path_resultsfolder + "pc_allpatientsepoch0_" + metainfo_str + "_" + repr(pc_n1+1) + "and" + repr(pc_n2+1) +".png")

def plot_pcvalues_2d_metacat(X_reduced, pc_n1, pc_n2, metainfo_str, metainfo_list, axisscale_fixed=True):
    """
    Scatter plot of PC values colored by patient group (categorical).
    """

    fig, ax1 = plt.subplots(1, 1)
    groups_unique = sorted(set(metainfo_list))
    
    # Choix automatique d'une palette qualitative
    cmap = plt.get_cmap("tab10")  # bon pour <=10 groupes
    color_dict = {g: cmap(i % 10) for i, g in enumerate(groups_unique)}

    # Plot group by group
    for g in groups_unique:
        indices = [i for i, grp in enumerate(metainfo_list) if grp == g]
        
        ax1.scatter(
            X_reduced[indices, pc_n1],
            X_reduced[indices, pc_n2],
            s=40,
            color=color_dict[g],
            label=g
        )

    ax1.set(
        title=f"Principal Components {pc_n1+1} and {pc_n2+1} and correlation with patient " + metainfo_str,
        xlabel=f"Principal Component {pc_n1+1}",
        ylabel=f"Principal Component {pc_n2+1}"
    )

    ax1.legend(title = metainfo_str)

    # ---- Axis scaling (same logic as yours)
    if axisscale_fixed:
        axis1, axis2 = 0, 1
    else:
        axis1, axis2 = pc_n1, pc_n2

    max_x = max(abs(X_reduced[:, axis1].min()), abs(X_reduced[:, axis1].max()))
    max_y = max(abs(X_reduced[:, axis2].min()), abs(X_reduced[:, axis2].max()))

    ax1.set_xticks(np.linspace(-max_x, max_x, 7))
    ax1.set_yticks(np.linspace(-max_y, max_y, 7))

    plt.savefig(path_resultsfolder + "pc_allpatientsepoch0_" + metainfo_str + "_" + repr(pc_n1+1) + "and" + repr(pc_n2+1) +".png")

