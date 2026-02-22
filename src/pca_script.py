# src/pca_script.py
"""
Scripts to perform PCA
"""

from sklearn.decomposition import PCA

from paths import *
import importdata_functions as idf
import pca_functions as pf 
import visualizeMRI_functions as vmf

import numpy as np 
import nibabel as nib # to get the nii format

# USER ACTION:
# Data to work from 
datatype_tochoose_1 = "testing/" # or "training/"
patient_name_1 = "patient104"
# PCA 
max_pc_calc = 100
# Explained variance infos and principal components in eigenbase
do_explainedvariance = False 
do_pcineigenbase= False 
pc_n1, pc_n2 = 8,9
# Reconstruction 
do_reconstruction = False
n_pc_toreconstruct = 5
epochs_to_plot = 3
# Eigenvector plot 
do_eigenvecplot = False 


# Extract nii file and convert it to Numpy array
nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, "4d", print_infos=False)
data_array = idf.convert_nii_file(nii_obj)

#1 PCA temporal: each 3D image as a sample (how voxels co-vary over time, temporal dynamics) -> 30 lines, >100 000 columns (dimensions).

# PCA CALCULATION 
# Prepare data
X = pf.pca1_transpose(data_array, print_infos=False) # to put temporal dimension t as 1st dimension and everything else in second dimension
X_scaled, scaler, removed_features_mask = pf.pca1_normalize(X) # 
# Do PCA 
pca1 = PCA(n_components=min(nii_obj.shape[3], max_pc_calc))  # Get all principal components (all t, since it limits in this case) or less if asked
X_reduced = pca1.fit_transform(X_scaled)
# Explained variance ratio: print + plot
if do_explainedvariance:
    print("Explained variance ratio:", pca1.explained_variance_ratio_)
    pf.plot_pca_explipower(pca1, patient_name_1)
# Plot pc values in eigenvector base 
if do_pcineigenbase:
    pf.plot_pcvalues_2d(X_reduced, pc_n1, pc_n2, patient_name_1, "_pc_in_eigenbase") #axisscale_fixed=False

# IMAGE RECONSTRUCTION
if do_reconstruction:
    # Reconstruct image - and residuals - using only n components
    X_reconstructed = pf.pca1_reconstruct(X_reduced, X, pca1, n_pc_toreconstruct, scaler, removed_features_mask)
    X_residuals = X - X_reconstructed
    # Re-shapping reconstructed data into original nii format 
    nii_reconstructed = pf.pca1_reformat(X_reconstructed, data_array, nii_obj, patient_name_1, n_pc_toreconstruct)
    niires_reconstructed = pf.pca1_reformat(X_residuals, data_array, nii_obj, patient_name_1, n_pc_toreconstruct)
    # Plot reconstruction and residuals (and original image for reference)
    vmf.plot_allepochs(nii_obj, patient_name_1, epoch_limit=epochs_to_plot, suffix = "_original_")
    vmf.plot_allepochs(nii_reconstructed, patient_name_1, epoch_limit=epochs_to_plot, suffix = "_" + repr(n_pc_toreconstruct) + "dims_")
    vmf.plot_allepochs(niires_reconstructed, patient_name_1, epoch_limit=epochs_to_plot, suffix = "_" + repr(n_pc_toreconstruct) + "dims_residuals")

# EIGENVECTOR PLOT 
if do_eigenvecplot:
    # Image 0 for reference
    vmf.plot_allepochs(nii_obj, patient_name_1, epoch_limit=1, suffix = "_for_reference")
    # Mean image 
    mean_image = np.mean(X, axis=0)
    vmf.from_longvec_to_image(mean_image, data_array.shape[:-1], nii_obj, patient_name_1, "_mean_image")
    # Every eigenvector
    for n_eigen in range(min(nii_obj.shape[3], max_pc_calc)):
    # n_eigen_toplot = 1 # must be < n_pca
        eigenvector_full = np.zeros((1, X.shape[1]))
        eigenvector_full[:, removed_features_mask] = pca1.components_[n_eigen, :]
        vmf.from_longvec_to_image(eigenvector_full, data_array.shape[:-1], nii_obj, patient_name_1, f"_eigenvector_{n_eigen+1}")

    # eigenvector = pca1.components_[n_eigen_toplot, :]
    # eigenvector_3d = eigenvector_full.reshape(data_array.shape[:-1]) # reshape into 3D according to the dimensions of the 3D image
    # nii_eigenvector = nib.Nifti1Image(eigenvector_3d, nii_obj.affine, nii_obj.header)
    # vmf.plot_oneepoch(nii_eigenvector, patient_name_1, suffix=f"_eigenvector_{n_eigen_toplot}")
    # vmf.plot_allepochs(nii_eigenvector, patient_name_1, epoch_limit=0, 