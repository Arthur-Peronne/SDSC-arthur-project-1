# src/pca_eachpatient.py
"""
Script to perform PCA with images from each patient (spatial PCA, not temporal anymore)
"""

import glob
from joblib import Parallel, delayed
import nibabel as nib # to import the files
import numpy as np
from sklearn.decomposition import PCA
import joblib

from paths import *
import importdata_functions as idf
import image_treatment as igt 
import pca_functions as pf 
import visualizeMRI_functions as vmf 

# USER ACTION:
timeframe = 0
reference_path = path_datadir + "training/patient001/patient001_4d.nii.gz" # For reference
reload_files = False
max_pc_calc = 10000 
recalculate_pca = False
do_explainedvariance = False 
do_pcineigenbase = False 
pc_n1, pc_n2 = 8,9
do_eigenvecplot = False
do_reconstruction = False 
n_pc_toreconstruct = 150
patienttoreconstruct_type = "training/" #  "testing/" or "training/"
patienttoreconstruct_number = "patient002"
patienttoreconstruct_index = 1 # should be number -1
correlations = True 

# LOAD IMAGE DATA
if reload_files:
    # Get all paths to nii files
    training_files = glob.glob(path_datadir + "training/patient*/patient*_4d.nii.gz")
    testing_files = glob.glob(path_datadir + "testing/patient*/patient*_4d.nii.gz")
    all_files = training_files + testing_files # and optional: all_files.sort() 
    # Create ref obj for resampling 
    nii_ref = nib.load(reference_path)
    # Epoch 0 extraction from the nii file to be searched from all_files, and resampling (256x256x10)
    def load_and_extract_epoch_zero(file_path, timeframe=0):
        # Load the NIfTI file
        nii_obj = nib.load(file_path)
        # Extract t0 data and create a tempo nii_file with it
        epoch_data = np.asanyarray(nii_obj.dataobj[..., timeframe], dtype=np.float32)
        nii_obj_t = nib.Nifti1Image(epoch_data, nii_obj.affine, nii_obj.header)
        # Resample to put it in correct resolution
        # Return the numpy array 
        nii_obj_res = igt.resample_basic(nii_obj_t, nii_ref)
        return np.asanyarray(nii_obj_res.dataobj, dtype=np.float32)
    # Use parallel processing to load and extract epoch zero
    img_list = Parallel(n_jobs=-1)(delayed(load_and_extract_epoch_zero)(file, timeframe) for file in all_files)
    # for i, img in enumerate(img_list):
    #     print(f"Image {i}: {img.shape}")
    # Convert the list of 3D images to a single 4D NumPy array
    img_array = np.stack(img_list)
    # print(img_array.shape)  # Should be (150, depth, height, width)
    out_path = path_tempodata_folder  + "nparraydata_for_pcaeachpatient.npy"
    np.save(out_path, img_array)

else:
    out_path = path_tempodata_folder  + "nparraydata_for_pcaeachpatient.npy"
    img_array = np.load(out_path)
    # print(img_array.shape, img_array.dtype)

# PCA
if  recalculate_pca:
    # Prepare data
    X = img_array.reshape(img_array.shape[0], -1)  # (150, 256*256*10)
    # X = pf.pca1_transpose(img_array, lines = img_array.shape[0]) # get a vector with n lines (all patients, ~150) and >100000 columns
    X, scaler, removed_features_mask = pf.pca1_normalize(X) # 
    # Do PCA 
    pca2 = PCA(n_components=min(X.shape[0], max_pc_calc))  # Get all principal components (all t, since it limits in this case) or less if asked
    X_pca = pca2.fit_transform(X)
    # Save PCA results for future work 
    joblib.dump(pca2, path_tempodata_folder + "pca2.joblib", compress=3)
    np.save(path_tempodata_folder +  "pca2_X_pca.npy", X_pca)
    meta = {
        "n_patients": X.shape[0],
        "n_features": X.shape[1],
        "n_components": pca2.n_components_,
        "explained_variance_ratio_": pca2.explained_variance_ratio_,
    }
    joblib.dump(meta, path_tempodata_folder + "pca2_meta.joblib", compress=3)
    joblib.dump(scaler, path_tempodata_folder + "pca2_scaler.pkl")
    joblib.dump(removed_features_mask, path_tempodata_folder + "pca2_removed_features_mask.pkl")
else:
    # Load results 
    pca2 = joblib.load(path_tempodata_folder + "pca2.joblib")
    X_pca = np.load(path_tempodata_folder +  "pca2_X_pca.npy")     # tes plots partent souvent de ça
    meta = joblib.load(path_tempodata_folder + "pca2_meta.joblib")
    scaler = joblib.load(path_tempodata_folder + "pca2_scaler.pkl")
    removed_features_mask = joblib.load(path_tempodata_folder + "pca2_removed_features_mask.pkl")

# EXPLAINED VARIANCE AND PLOT VALUES IN EIGENBASE
if do_explainedvariance:
    # print("Explained variance ratio:", pca2.explained_variance_ratio_)
    pf.plot_pca_explipower(pca2, "allpatients_epoch0")
# # Plot pc values in eigenvector base 
if do_pcineigenbase:
    pf.plot_pcvalues_2d(X_pca, pc_n1, pc_n2, "allpatients_epoch0", "_pc_in_eigenbase", scale_str ='Patient number', segments=False) #axisscale_fixed=False

# PLOT EIGENVECTORS
if do_eigenvecplot:
    X = img_array.reshape(img_array.shape[0], -1)  # (150, 256*256*10)
    # X = pf.pca1_transpose(img_array, lines = img_array.shape[0]) # get a vector with n lines (all patients, ~150) and >100000 columns
    # Image  reference (patient 001 here) for ref
    nii_ref = nib.load(reference_path)
    vmf.plot_allepochs(nii_ref, "patient001", epoch_limit=1, suffix = "_for_reference")
    # Mean image 
    mean_image = np.mean(X, axis=0)
    vmf.from_longvec_to_image(mean_image, img_array.shape[1:], nii_ref, "allpatients_epoch0", "_mean_image")
    #First 10 eigenvectors
    for n_eigen in range(10):
    # n_eigen_toplot = 1 # must be < n_pca
        eigenvector_full = np.zeros((1, X.shape[1]))
        eigenvector_full[:, removed_features_mask] = pca2.components_[n_eigen, :]
        vmf.from_longvec_to_image(eigenvector_full, img_array.shape[1:], nii_ref, "allpatients_epoch0", f"_eigenvector_{n_eigen+1}")

# IMAGE RECONSTRUCTION
if do_reconstruction:
    # Reconstruct image - and residuals - using only n components
    X = img_array.reshape(img_array.shape[0], -1)  # (150, 256*256*10)
    X_reconstructed = pf.pca1_reconstruct(X_pca, X, pca2, n_pc_toreconstruct, scaler, removed_features_mask)
    X_residuals = X - X_reconstructed
    # Re-shapping reconstructed data into original nii format 
    patient_path = path_datadir + patienttoreconstruct_type + patienttoreconstruct_number + "/" +  patienttoreconstruct_number + "_4d.nii.gz"  # Original nii file, to get info for reconstruction
    nii_original = nib.load(patient_path)
    epoch_data = np.asanyarray(nii_original.dataobj[..., timeframe], dtype=np.float32)
    nii_original_0 = nib.Nifti1Image(epoch_data, nii_original.affine, nii_original.header)
    nii_reconstructed = pf.pca2_reformat(X_reconstructed, img_array, nii_original, patienttoreconstruct_index)
    nii_residuals= pf.pca2_reformat(X_residuals, img_array, nii_original, patienttoreconstruct_index)
    # Plot reconstruction and residuals (and original image for reference)
    vmf.plot_oneepoch(nii_original_0, patienttoreconstruct_number, epoch_str = "epoch_0", details_str= "_original", cl_yesno = True)
    vmf.plot_oneepoch(nii_reconstructed, patienttoreconstruct_number, epoch_str = "epoch_0", details_str= "_" + repr(n_pc_toreconstruct) +"pc_reconstructed", cl_yesno = True)
    vmf.plot_oneepoch(nii_residuals, patienttoreconstruct_number, epoch_str = "epoch_0", details_str= "_" + repr(n_pc_toreconstruct) +"pc_residuals", cl_yesno = True)

# CORRELATIONS WITH PATIENT METADATA
if correlations:
    # Get all paths to Info files
    training_files = glob.glob(path_datadir + "training/patient*/Info.cfg")
    testing_files = glob.glob(path_datadir + "testing/patient*/Info.cfg")
    all_files = training_files + testing_files # and optional: all_files.sort() 
    # Get list of info values
    group_list, height_list, weight_list = [], [], []
    for file in all_files:
        dic = idf.read_info_cfg(file)
        group_list.append(dic["Group"])
        # height_list.append(dic["Height"])
        # weight_list.append(dic["Weight"])
    # pf.plot_pcvalues_2d_meta(X_pca, pc_n1, pc_n2, "Height", height_list)
    # pf.plot_pcvalues_2d_meta(X_pca, pc_n1, pc_n2, "Weight", weight_list)
    pf.plot_pcvalues_2d_metacat(X_pca, pc_n1, pc_n2, "Group", group_list)

# print(all_files)
# dic = idf.read_info_cfg(all_files[0])
# print(dic)
# print(all_files)