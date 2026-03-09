# src/resampling_functions.py
"""
Functions defined for the image registration
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path

from paths import *
import importdata_functions as idf 

def _axis_signs_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """
    Extract axis signs (+1/-1) from a nibabel affine.
    We use the diagonal as a proxy because your affines are of the form diag(±1,±1,±1,1).
    If a diagonal entry is 0 (unexpected), fall back to +1.
    """
    diag = np.diag(affine)[:3].astype(float)
    signs = []
    for v in diag:
        if v > 0:
            signs.append(1.0)
        elif v < 0:
            signs.append(-1.0)
        else:
            signs.append(1.0)
    return (signs[0], signs[1], signs[2])

def _make_sitk_from_nib(nib_img):
    """Convert nibabel image to SimpleITK image, forcing clean geometry."""
    data = nib_img.get_fdata(dtype=np.float32)  # for frame (intensity)
    zooms = nib_img.header.get_zooms()[:3]      # (sx, sy, sz) in mm

    # Nibabel array is (X,Y,Z). SimpleITK expects (Z,Y,X) in GetImageFromArray.
    data_zyx = np.transpose(data, (2, 1, 0))
    # print(data.shape)
    # print(data_zyx.shape)

    img = sitk.GetImageFromArray(data_zyx)

    img.SetSpacing(tuple(float(z) for z in zooms))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0))
    return img

def _resample_sitk(img, target_spacing, is_label=False):
    """Resample a SimpleITK image to target spacing."""
    orig_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    orig_size = np.array(img.GetSize(), dtype=np.int64)

    target_spacing = np.array(target_spacing, dtype=np.float64)

    # Keep same physical size: new_size ≈ old_size * old_spacing / new_spacing
    new_size = np.round(orig_size * (orig_spacing / target_spacing)).astype(int)
    new_size = [int(x) for x in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())  # identity

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)

    return resampler.Execute(img)

def _save_sitk_as_nifti(img, out_path, dtype=np.float32, axis_signs=(1.0, 1.0, 1.0)):
    """Save SimpleITK image as NIfTI via nibabel with clean affine."""
    arr_zyx = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0)).astype(dtype) # Transpose back dimensions

    sx, sy, sz = img.GetSpacing()
    ox, oy, oz = img.GetOrigin()

    sign_x, sign_y, sign_z = axis_signs

    affine = np.array([[sign_x * sx, 0,            0,            ox],
                       [0,            sign_y * sy, 0,            oy],
                       [0,            0,            sign_z * sz, oz],
                       [0,            0,            0,            1]], dtype=np.float64)

    nii = nib.Nifti1Image(arr_xyz, affine)
    # affine = np.array([[sx, 0,  0,  ox],
    #                    [0,  sy, 0,  oy],
    #                    [0,  0,  sz, oz],
    #                    [0,  0,  0,  1]], dtype=np.float64)

    # nii = nib.Nifti1Image(arr_xyz, affine)
    nii.header.set_zooms((sx, sy, sz))
    nib.save(nii, out_path)

    return nii

def resample_nifti_file(in_path, out_path, target_spacing, is_label=False, preserve_axis_signs=True):
    nib_img = nib.load(in_path)
    sitk_img = _make_sitk_from_nib(nib_img)

    # Minimal sign correction: keep only the diagonal signs from the original affine
    axis_signs = (1.0, 1.0, 1.0)
    if preserve_axis_signs:
        diag = np.diag(nib_img.affine)[:3].astype(float)
        axis_signs = tuple(-1.0 if v < 0 else 1.0 for v in diag)

    res = _resample_sitk(sitk_img, target_spacing, is_label=is_label)

    # For masks, preserve integer labels
    if is_label:
        nii_resampled = _save_sitk_as_nifti(res, out_path, dtype=np.uint8, axis_signs=axis_signs)
    else:
        nii_resampled = _save_sitk_as_nifti(res, out_path, dtype=np.float32, axis_signs=axis_signs)
    return nii_resampled

def print_spacing(in_path):
    img = nib.load(in_path)
    print(in_path, "zooms:", img.header.get_zooms()[:3], "shape:", img.shape)

def phys_size(img):
    shape = np.array(img.shape[:3], dtype=float)
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)
    return shape * zooms


def resample_all(target_spacing, only01 = True, limit = 1000):
    # Load all images and gt paths
    all_img, all_gt = idf.load_allframes(only01=only01)
    for i in range(min(len(all_img), limit)):
        # Load nii from paths, and crop images from gt masks
        nii_img= nib.load(all_img[i])
        nii_mask= nib.load(all_gt[i])
        # Create outpath
        path = Path(all_img[i])
        patient_id = path.parent.name
        frame_id = path.stem.split("_")[1].split(".")[0]
        save_path = path_tempodata_folder + "resampled_frames/" + patient_id +  "_" + frame_id + "_resampled"
        # Resample and save in tempo data
        nii_img_res = resample_nifti_file(all_img[i], save_path + ".nii.gz", target_spacing, is_label=False, preserve_axis_signs=True)
        nii_mask_res = resample_nifti_file(all_gt[i], save_path + "_gt.nii.gz", target_spacing, is_label=True, preserve_axis_signs=True)