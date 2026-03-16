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

def _load_and_flatten_nii(path, binary_mask=False, image_roi_only=False, roi_mask_path=None):
    """
    Load one NIfTI file and flatten it into a 1D vector.

    Parameters
    ----------
    path : str
        Path to the image or mask.
    binary_mask : bool
        If True, binarize the loaded data (for mask PCA).
    image_roi_only : bool
        If True, assume `path` is an image and keep intensities only where
        the ROI mask is non-zero. Outside ROI, values are set to 0.
    roi_mask_path : str or None
        Path to the ROI mask used when image_roi_only=True.
    """
    data = nib.load(path).get_fdata()

    # Case 1: PCA on masks
    if binary_mask:
        data = (data > 0).astype(np.float32)
    # Case 2: PCA on images restricted to ROI mask
    elif image_roi_only:
        if roi_mask_path is None:
            raise ValueError("roi_mask_path must be provided when image_roi_only=True")
        roi_mask = nib.load(roi_mask_path).get_fdata()
        roi_mask_bin = (roi_mask > 0)
        if data.shape != roi_mask_bin.shape:
            raise ValueError(
                f"Shape mismatch between image and ROI mask:\n"
                f"image: {path} -> {data.shape}\n"
                f"mask : {roi_mask_path} -> {roi_mask_bin.shape}"
            )
        data = data.astype(np.float32)
        data[~roi_mask_bin] = 0.0
    # Case 3: PCA on full images
    else:
        if np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        else:
            data = data.astype(np.float32)

    return data.ravel()

def get_vectorsarray(source_folder, pca_folder, recalculate = False, mask=False, binary_mask=False, image_roi_only=False, details_str = "", n_jobs=-1):
    """
    """

    # Folder to save/load
    save_suffix = ""
    if mask:
        save_suffix += "_gt"
    if binary_mask:
        save_suffix += "_bin"
    if image_roi_only:
        save_suffix += "_imgROIonly"

    out_path = (path_tempodata_folder + pca_folder + details_str + "_nparraydata_vectors" + save_suffix + ".npy")
    if source_folder == "registered_frames" or source_folder == "registered_framesBIS":
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
        # print(nii_paths)
        if len(nii_paths) == 0:
            raise ValueError(path_tempodata_folder + source_folder + "/patient*_frame*_" + suffix + "_gt.nii.gz")
        print("First file:", nii_paths[0])
        print("Last file :", nii_paths[-1])
        print("Total     :", len(nii_paths))
        # Parallel loading + flattening
        if image_roi_only:
            # For each image, use the corresponding GT mask as ROI
            roi_mask_paths = [
                p.replace(f"_{suffix}.nii.gz", f"_{suffix}_gt.nii.gz")
                for p in nii_paths
            ]
            data_list = Parallel(n_jobs=n_jobs)(delayed(_load_and_flatten_nii)(img_path, binary_mask=False, image_roi_only=True, roi_mask_path=mask_path)
                for img_path, mask_path in zip(nii_paths, roi_mask_paths)
            )
        else:
            data_list = Parallel(n_jobs=n_jobs)(delayed(_load_and_flatten_nii)(path, binary_mask=binary_mask, image_roi_only=False, roi_mask_path=None)
                for path in nii_paths
            )
        # data_list = Parallel(n_jobs=n_jobs)(delayed(_load_and_flatten_nii)(path, binary_mask=binary_mask) for path in nii_paths)
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

def patient_metalists(all_files, returnonlyone = False, whichtoreturn = "group"):
    """
    """
    group_list, height_list, weight_list = [], [], []

    for file in all_files:
        dic = idf.read_info_cfg(file)
        group_list.append(dic["Group"])
        height_list.append(dic["Height"])
        weight_list.append(dic["Weight"])

    if returnonlyone:
        if whichtoreturn == "group":
            return group_list
        elif whichtoreturn == "height":
            return height_list
        elif whichtoreturn == "weight":
            return weight_list
        else:
            print("WARNING!!! Wrong whichtoreturn name in patient_metalists function! group_list returned by default")
            return group_list
    else:
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
