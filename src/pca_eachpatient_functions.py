# src/pca_eachpatient_functions.py
"""
Script for the functions for the PCA "each patient"
"""

import glob
import numpy as np
import nibabel as nib
import joblib
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from paths import * 
import visualizeMRI_functions as vmf 
import importdata_functions as idf 
import pca_functions as pf 

def _load_and_flatten_nii(path, binary_mask=False):
    """
    Load one NIfTI file and flatten it into a 1D vector.
    """
    data = nib.load(path).get_fdata()

    # Optional dtype reduction to save memory
    if binary_mask:
        data = (data > 0).astype(np.float32)
        # data = (data > 0).astype(np.uint8)
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    return data.ravel()

def get_vectorsarray(source_folder, pca_folder, recalculate = False, mask=False, binary_mask=False, details_str = "", n_jobs=-1):
    """
    """

    # Folder to save/load
    save_suffix = "_gt" if mask else ""
    if binary_mask:
        save_suffix += "_bin"
    out_path = path_tempodata_folder + pca_folder + details_str + "_nparraydata_vectors_" + save_suffix + ".npy"
    if source_folder == "registered_frames":
        suffix = "registered"
    else:
        suffix = "cropped"
    if recalculate: # Load all nii and extract data and save vector array
        # Build input paths
        if not mask:
            nii_paths = sorted(
                glob.glob(path_tempodata_folder + source_folder + "/patient*_frame*_" + suffix + ".nii.gz")
            )
        else:
            nii_paths = sorted(
                glob.glob(path_tempodata_folder + source_folder + "/patient*_frame*_" + suffix + "_gt.nii.gz")
            )
        print(nii_paths)
        if len(nii_paths) == 0:
            raise ValueError(path_tempodata_folder + source_folder + "/patient*_frame*_" + suffix + "_gt.nii.gz")
        print("First file:", nii_paths[0])
        print("Last file :", nii_paths[-1])
        print("Total     :", len(nii_paths))
        # Parallel loading + flattening
        data_list = Parallel(n_jobs=n_jobs)(delayed(_load_and_flatten_nii)(path, binary_mask=binary_mask) for path in nii_paths)
        # Stack into shape (n_patients, n_voxels)
        data_array = np.stack(data_list, axis=0)
        np.save(out_path, data_array)
    else: # Load directly array previously saved 
        data_array = np.load(out_path)

    return data_array

def pca_patients(X, pca_folder, pca_description,  normalize_rows=True, recalculate = False, max_pc_calc = 1000):
    """
    """
    if recalculate:
        if normalize_rows:
            X -= X.mean(axis=1, keepdims=True)
        pca = PCA(n_components=min(X.shape[0], max_pc_calc))  # Get all principal components (all t, since it limits in this case) or less if asked
        X_pca = pca.fit_transform(X)
        joblib.dump(pca, path_tempodata_folder + pca_folder +  pca_description + "_pca.joblib", compress=3)
        np.save(path_tempodata_folder + pca_folder +  pca_description + "_X_pca.npy", X_pca)
        meta = {
            "n_patients": X.shape[0],
            "n_features": X.shape[1],
            "n_components": pca.n_components_,
            "explained_variance_ratio_": pca.explained_variance_ratio_,
        }
        joblib.dump(meta, path_tempodata_folder + pca_folder + pca_description + "_meta.joblib", compress=3)

    else:
        # Load results 
        pca =joblib.load(path_tempodata_folder + pca_folder +  pca_description + "_pca.joblib")
        X_pca = np.load(path_tempodata_folder + pca_folder +  pca_description + "_X_pca.npy")
        meta = joblib.load(path_tempodata_folder + pca_folder + pca_description + "_meta.joblib")

    return pca, X_pca, meta

def plot_eigenvectors(X, pca, original_shape, pca_description, eigenvectors_toplot=10):
    # PLOT EIGENVECTORS
    X_4d = X.reshape((X.shape[0],) + original_shape)
    # Plot reference
    nii_ref = nib.load(path_tempodata_folder + "cropped_frames/patient001_frame01_cropped.nii.gz")
    vmf.plot_oneimg(nii_ref, patient_str = pca_description + "_patient001", file_str ="frame001", details_str="ORIGINAL")
    # Plot mean image 
    mean_img = X_4d.mean(axis=0)
    nii_mean = nib.Nifti1Image(mean_img, nii_ref.affine, nii_ref.header)
    vmf.plot_oneimg(nii_mean, patient_str =pca_description, file_str ="frame001", details_str="mean_image")
    # Plot first eigenvectors
    for n_eigen in range(eigenvectors_toplot):
        eigenvector = pca.components_[n_eigen, :]
        eigenvector_3D = eigenvector.reshape(original_shape)
        eigenvector_nii = nib.Nifti1Image(eigenvector_3D, nii_ref.affine, nii_ref.header)
        vmf.plot_oneimg(eigenvector_nii, patient_str =pca_description, file_str ="frame001", details_str=f"_eigenvector_{n_eigen+1}")

def patient_metalists(all_files):
    """
    """
    group_list, height_list, weight_list = [], [], []

    for file in all_files:
        dic = idf.read_info_cfg(file)
        group_list.append(dic["Group"])
        height_list.append(dic["Height"])
        weight_list.append(dic["Weight"])

    return group_list, height_list, weight_list 

def plot_pca_patientmeta(X_pca, pc_n1, pc_n2):
    """
    """
    # Get paths
    all_files = idf.import_patientmetapaths(printinfos=False)
    # Extract infos in dic 
    group_list, height_list, weight_list  = patient_metalists(all_files)
    # Plot 
    pf.plot_pcvalues_2d_meta(X_pca, pc_n1, pc_n2, "Height", height_list)
    pf.plot_pcvalues_2d_meta(X_pca, pc_n1, pc_n2, "Weight", weight_list)
    pf.plot_pcvalues_2d_metacat(X_pca, pc_n1, pc_n2, "Group", group_list)
