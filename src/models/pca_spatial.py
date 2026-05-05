# src/models/pca_spatial.py
"""
Script for the functions for the PCA "each patient"
"""

import glob
import numpy as np
import nibabel as nib
import joblib
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from src.config import TEMPODATA_FOLDER
from src.visualization import mri_plots as mrp
from src.data import importdata as ipd
from src.models import pca as pf

def _load_and_flatten_nii(path, binary_mask=False, image_roi_only=False, roi_mask_path=None, flatten=True):
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
        data = data.astype(np.float32)

    if flatten:
        return data.ravel()
    else:
        return data


def get_vectorsarray(
    source_folder,
    pca_folder,
    recalculate=False,
    mask=False,
    binary_mask=False,
    image_roi_only=False,
    flatten=True,
    n_jobs=-1,
    frame_type="ED",
):
    """
    Load or compute the array of patient images.
 
    Parameters
    ----------
    source_folder : str
        Subfolder in TEMPODATA_FOLDER containing the registered NIfTI files.
    pca_folder : str
        Subfolder in TEMPODATA_FOLDER where the .npy array is saved/loaded.
    recalculate : bool
        If True, reload NIfTI files and recompute the array.
    mask : bool
        If True, load GT mask files instead of images.
    binary_mask : bool
        If True, binarize the loaded data.
    image_roi_only : bool
        If True, zero out voxels outside the ROI mask.
    flatten : bool
        If True, flatten each image to 1D. If False, keep 3D shape.
    n_jobs : int
        Number of parallel jobs for loading.
    frame_type : str
        "ED" → ED frames (frame01 for all, frame04 for patient090)
        "ES" → ES frames (the other frame for each patient)
        Passed to load_allframes_registered.
 
    Returns
    -------
    data_array : np.ndarray
        Shape (n_patients, n_voxels) if flatten=True,
              (n_patients, H, W, D)  if flatten=False.
    """
    from src.data import importdata as ipd
 
    # Build save suffix
    save_suffix = ""
    if mask:
        save_suffix += "_gt"
    if binary_mask:
        save_suffix += "_bin"
    if image_roi_only:
        save_suffix += "_imgROIonly"
    save_suffix += f"_{frame_type}"
    if not flatten:
        save_suffix += "_4d"
    else:
        save_suffix += "_flat3d"
 
    out_path = TEMPODATA_FOLDER / pca_folder / f"Xvectors{save_suffix}.npy"
 
    if recalculate:
        # Get paths via importdata — patient090 exception handled there
        all_img, all_gt = ipd.load_allframes_registered(
            folder=source_folder,
            frame_type=frame_type,
        )
 
        if mask:
            nii_paths = all_gt
        else:
            nii_paths = all_img
 
        if len(nii_paths) == 0:
            raise ValueError(
                f"No NIfTI files found in {source_folder!r} "
                f"with frame_type={frame_type!r}"
            )
 
        print("First file:", nii_paths[0])
        print("Last file :", nii_paths[-1])
        print("Total     :", len(nii_paths))
 
        # Parallel loading
        if image_roi_only:
            roi_mask_paths = [
                p.replace("_registered.nii.gz", "_registered_gt.nii.gz")
                for p in nii_paths
            ]
            data_list = Parallel(n_jobs=n_jobs)(
                delayed(_load_and_flatten_nii)(
                    img_path, binary_mask=False, image_roi_only=True,
                    roi_mask_path=mask_path, flatten=flatten
                )
                for img_path, mask_path in zip(nii_paths, roi_mask_paths)
            )
        else:
            data_list = Parallel(n_jobs=n_jobs)(
                delayed(_load_and_flatten_nii)(
                    path, binary_mask=binary_mask, image_roi_only=False,
                    roi_mask_path=None, flatten=flatten
                )
                for path in nii_paths
            )
 
        data_array = np.stack(data_list, axis=0)
        np.save(out_path, data_array)
 
    else:
        data_array = np.load(out_path)
 
    return data_array


def pca_patients(X, pca_folder, pca_description, normalize_rows=True, recalculatePCA=False, max_pc_calc=1000, addstring=""):
    """
    """

    folder_name = TEMPODATA_FOLDER / pca_folder / f"{pca_description}"

    if recalculatePCA:
        if normalize_rows:
            X -= X.mean(axis=1, keepdims=True)
        pca = PCA(n_components=min(X.shape[0], max_pc_calc))
        X_pca = pca.fit_transform(X)
        # Save 
        folder_name.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, folder_name / f"_pca{addstring}.joblib", compress=3)
        np.save(folder_name / f"_X_pca{addstring}.npy", X_pca)
        meta = {
            "n_patients": X.shape[0],
            "n_features": X.shape[1],
            "n_components": pca.n_components_,
            "explained_variance_ratio_": pca.explained_variance_ratio_,
        }
        joblib.dump(meta, folder_name / f"_meta{addstring}.joblib", compress=3)

    else:
        pca = joblib.load(folder_name / f"_pca{addstring}.joblib")
        X_pca = np.load(folder_name / f"_X_pca{addstring}.npy")
        meta = joblib.load(folder_name / f"_meta{addstring}.joblib")

    return pca, X_pca, meta


def plot_eigenvectors(X, pca, original_shape, pca_description, eigenvectors_toplot=10):
    """
    """
    X_4d = X.reshape((X.shape[0],) + original_shape)
    nii_ref = nib.load(TEMPODATA_FOLDER / "cropped_frames/patient001_frame01_cropped.nii.gz")
    mrp.plot_oneimg(nii_ref, patient_str=f"{pca_description}_patient001", file_str="frame001", details_str="ORIGINAL")
    mean_img = X_4d.mean(axis=0)
    nii_mean = nib.Nifti1Image(mean_img, nii_ref.affine, nii_ref.header)
    mrp.plot_oneimg(nii_mean, patient_str=pca_description, file_str="frame001", details_str="mean_image")
    for n_eigen in range(eigenvectors_toplot):
        eigenvector = pca.components_[n_eigen, :]
        eigenvector_3D = eigenvector.reshape(original_shape)
        eigenvector_nii = nib.Nifti1Image(eigenvector_3D, nii_ref.affine, nii_ref.header)
        mrp.plot_oneimg(eigenvector_nii, patient_str=pca_description, file_str="frame001", details_str=f"_eigenvector_{n_eigen+1}")


def patient_metalists(all_files, returnonlyone=False, whichtoreturn="group"):
    """
    """
    group_list, height_list, weight_list = [], [], []

    for file in all_files:
        dic = ipd.read_info_cfg(file)
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
    all_files = ipd.import_patientmetapaths(printinfos=False)
    group_list, height_list, weight_list = patient_metalists(all_files)
    
    # Truncate metadata to match X_pca size
    n = X_pca.shape[0]
    group_list  = group_list[:n]
    height_list = height_list[:n]
    weight_list = weight_list[:n]
    
    pf.plot_pcvalues_2d_meta(X_pca, pc_n1, pc_n2, "Height", height_list)
    pf.plot_pcvalues_2d_meta(X_pca, pc_n1, pc_n2, "Weight", weight_list)
    pf.plot_pcvalues_2d_metacat(X_pca, pc_n1, pc_n2, "Group", group_list)