# src/pca_functions.py
"""
Scripts to perform PCA
"""

from sklearn.decomposition import PCA

from paths import *
import importdata_functions as idf
import pca_functions as pf 
import visualizeMRI_functions as vmf

import numpy as np
import nibabel as nib # to import the files
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import StandardScaler, VarianceThreshold

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "testing/" # or "training/"
patient_name_1 = "patient103"
n_pc_toreconstruct = 30
epochs_to_plot = 3

# Extract nii file and convert it to Numpy array
nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, "4d", print_infos=False)
data_array = idf.convert_nii_file(nii_obj)

# PCA 1: each 3D image as a sample (how voxels co-vary over time, temporal dynamics) -> 30 lines, >100 000 columns (dimensions).

# Prepare data
X = pf.pca1_transpose(data_array, print_infos=False)
# X_scaled = pf.pca_clean(X)
selector = VarianceThreshold()
X_filtered = selector.fit_transform(X)
removed_features_mask = selector.get_support(indices=False)  # Mask of features that were NOT removed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
# PCA 
pca1 = PCA(n_components=nii_obj.shape[3])  # Get all principal components (all t, since it limits in this case)
X_reduced = pca1.fit_transform(X_scaled)
# Explained variance ratio: print + plot
# print("Explained variance ratio:", pca1.explained_variance_ratio_)
pf.plot_pca_explipower(pca1, patient_name_1)

# Reconstruct image - and residuals - using only n components
X_reconstructed_scaled = X_reduced[:, :n_pc_toreconstruct] @ pca1.components_[:n_pc_toreconstruct, :] # transformed data (only first n columns, shape (30,n)) times eigen_vectors (only first n rows, shape(n,>100000) -> shape (30, >100000), OK!
X_reconstructed_filtered = scaler.inverse_transform(X_reconstructed_scaled)
X_reconstructed = np.zeros_like(X)
X_reconstructed[:, removed_features_mask] = X_reconstructed_filtered
removed_features_mean = np.mean(X[:, ~removed_features_mask], axis=0)
X_reconstructed[:, ~removed_features_mask] = np.tile(removed_features_mean, (X.shape[0], 1))
X_residuals = X - X_reconstructed
# Reshape reconstructed data
X_reconstructed_3d = X_reconstructed.reshape(data_array.shape[-1], *data_array.shape[:-1]) # Shape of the initial 4D data, but with epochs as first dimension
X_reconstructed_4d = np.transpose(X_reconstructed_3d, (1, 2, 3, 0))
nii_reconstructed = nib.Nifti1Image(X_reconstructed_4d, nii_obj.affine, nii_obj.header)
nib.save(nii_reconstructed, path_resultimagesfolder+ patient_name_1 + "_projected_" + repr(n_pc_toreconstruct) + "_4d.nii.gz")
# Reshape residuals
X_residuals_3d = X_residuals.reshape(data_array.shape[-1], *data_array.shape[:-1]) # Shape of the initial 4D data, but with epochs as first dimension
X_residuals_4d = np.transpose(X_residuals_3d, (1, 2, 3, 0))
niires_reconstructed = nib.Nifti1Image(X_residuals_4d, nii_obj.affine, nii_obj.header)
nib.save(niires_reconstructed, path_resultimagesfolder+ patient_name_1 + "_residuals_" + repr(n_pc_toreconstruct) + "_4d.nii.gz")
# Plot reconstruction and residuals 
vmf.plot_allepochs(nii_obj, patient_name_1, epoch_limit=epochs_to_plot, suffix = "_original_")
vmf.plot_allepochs(nii_reconstructed, patient_name_1, epoch_limit=epochs_to_plot, suffix = "_" + repr(n_pc_toreconstruct) + "dims_")
vmf.plot_allepochs(niires_reconstructed, patient_name_1, epoch_limit=epochs_to_plot, suffix = "_" + repr(n_pc_toreconstruct) + "dims_residuals")





# PCA 1 for every patient 

# For every 50 patients
# Get nii file and put in numpy array 
# Transpose and concatenate 
# Then same 



# PCA 2: each voxel as a sample (spatial modes, spatial patterns) -> >100 000 lines, 30 columns (dimensions).
