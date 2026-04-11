# src/data/imagetreatment.py
"""
Basic MRI image treatment functions (cropping, resampling).
"""

import nibabel as nib
from nilearn.image import resample_img
from nilearn.image import resample_to_img
import numpy as np
import glob
from pathlib import Path

from src.config import DATADIR, TEMPODATA_FOLDER, RESULTS_FOLDER
from src.visualization import mri_plots as mrp


def resample_basic(target_img, reference_img, patient_name_1,
                   epoch_toplot=0, target_shape=(256, 256, 10), savefile=False, plotimage=False):
    """
    Resample target_img to match reference_img affine and target_shape.
    """
    resampled_img = resample_img(
        target_img,
        target_affine=reference_img.affine,
        target_shape=target_shape,
        interpolation='linear')

    if savefile:
        resampled_path = RESULTS_FOLDER / f"{patient_name_1}_4d_RESAMPLED001.nii.gz"
        nib.save(resampled_img, resampled_path)

    if plotimage:
        target_img_1t = mrp.get_oneepoch(target_img, epoch_toplot)
        mrp.plot_oneepoch(target_img_1t, patient_name_1, epoch_str="_epoch0", details_str="")
        resampled_img_1t = mrp.get_oneepoch(resampled_img, epoch_toplot)
        mrp.plot_oneepoch(resampled_img_1t, patient_name_1, epoch_str="_epoch0", details_str="RESAMPLED001")

    return resampled_img


def crop_heartzone_oneimage(nii_img, nii_mask):
    """
    Crop image to cardiac mask region (set outside mask to NaN).
    """
    nii_mask_resampled = resample_to_img(nii_mask, nii_img, interpolation="nearest")
    data_img = nii_img.get_fdata()
    data_mask = nii_mask_resampled.get_fdata()
    region_to_keep = (data_mask != 0)
    data_cropped = data_img.copy()
    data_cropped[~region_to_keep] = np.nan
    nii_cropped = nib.Nifti1Image(data_cropped, affine=nii_img.affine)
    return nii_cropped


def loaddata_tocrop():
    """
    Load paths to all frame NIfTI files and their GT masks.
    """
    training_img = glob.glob(str(DATADIR / "training/patient*/patient*_frame[0-9][0-9].nii.gz"))
    testing_img = glob.glob(str(DATADIR / "testing/patient*/patient*_frame[0-9][0-9].nii.gz"))
    all_img = training_img + testing_img

    training_gt = glob.glob(str(DATADIR / "training/patient*/patient*_frame*_gt.nii.gz"))
    testing_gt = glob.glob(str(DATADIR / "testing/patient*/patient*_frame*_gt.nii.gz"))
    all_gt = training_gt + testing_gt

    return all_img, all_gt


def crop_heartzone_allpatients(limit=1000):
    """
    Crop all frames to cardiac mask region and save results.
    """
    all_img, all_gt = loaddata_tocrop()
    for i in range(min(len(all_img), limit)):
        nii_img = nib.load(all_img[i])
        nii_mask = nib.load(all_gt[i])
        nii_cropped = crop_heartzone_oneimage(nii_img, nii_mask)
        path = Path(all_img[i])
        patient_id = path.parent.name
        frame_id = path.stem.split("_")[1]
        save_path = TEMPODATA_FOLDER / "cropped_nii" / f"{patient_id}_{frame_id}_cropped.nii.gz"
        nib.save(nii_cropped, save_path)


def plot_cropped_files(limit=10):
    """
    Plot and save cropped NIfTI files.
    """
    cropped_nii_paths = glob.glob(str(TEMPODATA_FOLDER / "cropped_nii/*.nii.gz"))
    for i, nii_path in enumerate(cropped_nii_paths):
        nii_cropped = nib.load(nii_path)
        path = Path(nii_path)
        stem = Path(path.stem).stem
        parts = stem.split("_")
        patient_from_filename = parts[0]
        frame_id = parts[1]
        mrp.plot_oneepoch(nii_cropped, patient_from_filename, oldstyle=False,
                          epoch_str=f"_{frame_id}_", details_str="cropped")
        if i > limit:
            break


def bbox_from_masked_nan(img, filtered="zeros"):
    """
    Compute bounding box from a masked image (NaN or zeros outside ROI).

    Parameters
    ----------
    img : nib.Nifti1Image
    filtered : str
        "zeros" or "nans"

    Returns
    -------
    bbox : tuple (xmin, xmax, ymin, ymax, zmin, zmax)
    sizes : tuple (Nx, Ny, Nz)
    """
    data_3d = img.get_fdata()

    if filtered == "zeros":
        roi = (data_3d > 0)
    else:
        roi = np.isfinite(data_3d)

    coords = np.where(roi)
    xmin, xmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    zmin, zmax = coords[2].min(), coords[2].max()

    Nx = xmax - xmin + 1
    Ny = ymax - ymin + 1
    Nz = zmax - zmin + 1

    return (xmin, xmax, ymin, ymax, zmin, zmax), (Nx, Ny, Nz)