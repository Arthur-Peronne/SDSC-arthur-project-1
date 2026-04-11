# src/data/registration.py
"""
Rigid registration of cardiac MRI frames and Dice-based evaluation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import glob

from src.config import TEMPODATA_FOLDER
from src.data import importdata as ipd
from src.data import resampling as rsp
from src.data.geometry import crop_to_reference_window


def _sitk_to_nib(img_sitk, reference_nib, dtype=np.float32):
    """
    Convert SimpleITK image back to nibabel NIfTI, using the reference affine.
    """
    arr_zyx = sitk.GetArrayFromImage(img_sitk)      # (Z,Y,X)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0)).astype(dtype)  # back to (X,Y,Z)

    nii = nib.Nifti1Image(arr_xyz, reference_nib.affine)
    nii.header.set_zooms(reference_nib.header.get_zooms()[:3])

    return nii


def rigid_register_one_patient(
    fixed_img_full_nii,
    fixed_mask_full_nii,
    fixed_mask_crop_nii,
    moving_mask_crop_nii,
    moving_img_full_nii,
    moving_mask_full_nii,
    number_of_iterations=200
):
    """
    Estimate rigid transform on cropped binary masks,
    then apply it to full resampled image and full multi-label mask,
    resampling onto the full fixed grid.
    """
    # --- SITK conversion ---
    fixed_img_full_sitk = rsp._make_sitk_from_nib(fixed_img_full_nii)
    fixed_mask_full_sitk = rsp._make_sitk_from_nib(fixed_mask_full_nii)

    fixed_mask_crop_sitk = rsp._make_sitk_from_nib(fixed_mask_crop_nii)
    moving_mask_crop_sitk = rsp._make_sitk_from_nib(moving_mask_crop_nii)

    moving_img_full_sitk = rsp._make_sitk_from_nib(moving_img_full_nii)
    moving_mask_full_sitk = rsp._make_sitk_from_nib(moving_mask_full_nii)

    # --- binary masks for registration ---
    fixed_mask_bin_sitk = sitk.Cast(fixed_mask_crop_sitk > 0, sitk.sitkFloat32)
    moving_mask_bin_sitk = sitk.Cast(moving_mask_crop_sitk > 0, sitk.sitkFloat32)

    # --- initialize rigid transform ---
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_mask_bin_sitk,
        moving_mask_bin_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # --- registration setup ---
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=number_of_iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # --- estimate transform on cropped binary masks ---
    final_transform = registration_method.Execute(
        fixed_mask_bin_sitk,
        moving_mask_bin_sitk
    )

    # --- apply transform to full image, resample onto fixed full grid ---
    registered_img_sitk = sitk.Resample(
        moving_img_full_sitk,
        fixed_img_full_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img_full_sitk.GetPixelID()
    )

    # --- apply transform to full multi-label mask, resample onto fixed full grid ---
    registered_mask_sitk = sitk.Resample(
        moving_mask_full_sitk,
        fixed_mask_full_sitk,
        final_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    # --- back to nibabel ---
    registered_img_nii = _sitk_to_nib(registered_img_sitk, fixed_img_full_nii, dtype=np.float32)
    registered_mask_nii = _sitk_to_nib(registered_mask_sitk, fixed_mask_full_nii, dtype=np.uint8)

    return registered_img_nii, registered_mask_nii, final_transform


def print_rigid_transform_info(transform):
    """
    Print parameters of a SimpleITK rigid transform.
    Works well for Euler3DTransform returned by rigid registration.
    """
    print("Transform type:", transform.GetName())

    params = transform.GetParameters()
    fixed_params = transform.GetFixedParameters()

    print("Parameters:", params)
    print("Fixed parameters:", fixed_params)

    if len(params) == 6:
        rx, ry, rz, tx, ty, tz = params

        print(f"Rotation X (rad): {rx:.6f}")
        print(f"Rotation Y (rad): {ry:.6f}")
        print(f"Rotation Z (rad): {rz:.6f}")

        print(f"Rotation X (deg): {np.degrees(rx):.3f}")
        print(f"Rotation Y (deg): {np.degrees(ry):.3f}")
        print(f"Rotation Z (deg): {np.degrees(rz):.3f}")

        print(f"Translation X: {tx:.3f}")
        print(f"Translation Y: {ty:.3f}")
        print(f"Translation Z: {tz:.3f}")


def register_all_frames(reference_patient="patient001", crop_after_registration=True, crop_size_after_registration=(128, 128, 32), limit=1000):
    """
    Register all patients to a fixed reference.

    Transform is estimated on cropped masks, applied on full resampled image/mask.
    Optionally crop afterwards to the exact spatial window of the fixed reference crop.
    """
    all_img_crop = sorted(ipd.load_allcroppedframes())

    out_folder = TEMPODATA_FOLDER / "registered_framesBIS"
    out_folder.mkdir(parents=True, exist_ok=True)

    # Find reference cropped image dynamically
    ref_img_crop_path = None
    for p in all_img_crop:
        if Path(p).name.startswith(reference_patient + "_"):
            ref_img_crop_path = p
            break
    if ref_img_crop_path is None:
        raise ValueError(f"Reference patient {reference_patient} not found.")

    ref_crop_path = Path(ref_img_crop_path)
    ref_patient_id, ref_frame_id = ref_crop_path.stem.split("_")[:2]

    fixed_img_crop_nii = nib.load(ref_img_crop_path)
    fixed_mask_crop_path = ref_img_crop_path.replace("_cropped.nii.gz", "_cropped_gt.nii.gz")
    fixed_mask_crop_nii = nib.load(fixed_mask_crop_path)

    fixed_img_full_path = TEMPODATA_FOLDER / "resampled_frames" / f"{ref_patient_id}_{ref_frame_id}_resampled.nii.gz"
    fixed_mask_full_path = TEMPODATA_FOLDER / "resampled_frames" / f"{ref_patient_id}_{ref_frame_id}_resampled_gt.nii.gz"

    fixed_img_full_nii = nib.load(fixed_img_full_path)
    fixed_mask_full_nii = nib.load(fixed_mask_full_path)

    if tuple(crop_size_after_registration) != fixed_img_crop_nii.shape:
        print("Warning: crop_size_after_registration differs from reference cropped image shape.")
        print("Requested:", crop_size_after_registration)
        print("Reference:", fixed_img_crop_nii.shape)

    n_files = min(len(all_img_crop), limit)

    for i in range(n_files):
        img_crop_path = all_img_crop[i]
        path = Path(img_crop_path)
        patient_id, frame_id = path.stem.split("_")[:2]

        img_out = out_folder / f"{patient_id}_{frame_id}_registered.nii.gz"
        mask_out = out_folder / f"{patient_id}_{frame_id}_registered_gt.nii.gz"

        # Skip reference patient
        if patient_id == ref_patient_id and frame_id == ref_frame_id:
            if crop_after_registration:
                nib.save(fixed_img_crop_nii, img_out)
                nib.save(fixed_mask_crop_nii, mask_out)
            else:
                nib.save(fixed_img_full_nii, img_out)
                nib.save(fixed_mask_full_nii, mask_out)
            print(f"[{i+1}/{n_files}] Saved reference {img_out.name}")
            continue

        moving_mask_crop_path = str(path).replace("_cropped.nii.gz", "_cropped_gt.nii.gz")
        moving_mask_crop_nii = nib.load(moving_mask_crop_path)

        moving_img_full_path = TEMPODATA_FOLDER / "resampled_frames" / f"{patient_id}_{frame_id}_resampled.nii.gz"
        moving_mask_full_path = TEMPODATA_FOLDER / "resampled_frames" / f"{patient_id}_{frame_id}_resampled_gt.nii.gz"

        moving_img_full_nii = nib.load(moving_img_full_path)
        moving_mask_full_nii = nib.load(moving_mask_full_path)

        reg_img_full_nii, reg_mask_full_nii, final_transform = rigid_register_one_patient(
            fixed_img_full_nii, fixed_mask_full_nii, fixed_img_crop_nii,
            moving_mask_crop_nii,
            moving_img_full_nii, moving_mask_full_nii
        )

        if crop_after_registration:
            reg_img_nii, reg_mask_nii = crop_to_reference_window(
                reg_img_full_nii, reg_mask_full_nii, fixed_img_crop_nii
            )
        else:
            reg_img_nii, reg_mask_nii = reg_img_full_nii, reg_mask_full_nii

        nib.save(reg_img_nii, img_out)
        nib.save(reg_mask_nii, mask_out)
        print(f"[{i+1}/{n_files}] Saved {img_out.name}")


def dice_score(mask1, mask2):
    """
    Dice score between two binary masks.
    """
    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    volume_sum = m1.sum() + m2.sum()

    if volume_sum == 0:
        return 1.0

    return 2.0 * intersection / volume_sum


def dice_all_patients(reference_patient="patient001", cropped_folder="cropped_frames", registered_folder="registered_framesBIS"):
    """
    Compute Dice score before/after registration for all patients.

    Dice before:
        cropped moving mask vs cropped reference mask
    Dice after:
        registered moving mask vs cropped reference mask

    Returns
    -------
    results_list : list of dict
        Per-patient dice results.
    """
    cropped_mask_paths = sorted(glob.glob(str(TEMPODATA_FOLDER / cropped_folder / "patient*_frame*_cropped_gt.nii.gz")))
    registered_mask_paths = sorted(glob.glob(str(TEMPODATA_FOLDER / registered_folder / "patient*_frame*_registered_gt.nii.gz")))

    ref_mask_path = None
    for p in cropped_mask_paths:
        if Path(p).name.startswith(reference_patient + "_"):
            ref_mask_path = p
            break
    if ref_mask_path is None:
        raise ValueError(f"Reference patient {reference_patient} not found in {cropped_folder}.")
    ref_mask = nib.load(ref_mask_path).get_fdata()

    n_files = min(len(cropped_mask_paths), len(registered_mask_paths))
    if len(cropped_mask_paths) != len(registered_mask_paths):
        print(f"Warning: different number of files.")
        print(f"cropped masks   : {len(cropped_mask_paths)}")
        print(f"registered masks: {len(registered_mask_paths)}")
        print(f"Using first {n_files} paired files after sorting.")

    results_list = []
    for i in range(n_files):
        cropped_mask_path = cropped_mask_paths[i]
        reg_mask_path = registered_mask_paths[i]
        cropped_name = Path(cropped_mask_path).name.replace("_cropped_gt.nii.gz", "")
        reg_name = Path(reg_mask_path).name.replace("_registered_gt.nii.gz", "")
        cropped_patient_id, cropped_frame_id = cropped_name.split("_")[:2]
        reg_patient_id, reg_frame_id = reg_name.split("_")[:2]

        if (cropped_patient_id != reg_patient_id) or (cropped_frame_id != reg_frame_id):
            raise ValueError(
                f"Mismatch between cropped and registered masks at index {i}:\n"
                f"cropped    : {cropped_mask_path}\n"
                f"registered : {reg_mask_path}"
            )

        patient_id = reg_patient_id
        frame_id = reg_frame_id
        cropped_mask = nib.load(cropped_mask_path).get_fdata()
        reg_mask = nib.load(reg_mask_path).get_fdata()

        dice_before = dice_score(ref_mask, cropped_mask)
        dice_after = dice_score(ref_mask, reg_mask)

        results_list.append({
            "patient_id": patient_id,
            "frame_id": frame_id,
            "dice_before": float(dice_before),
            "dice_after": float(dice_after),
            "dice_gain": float(dice_after - dice_before),
        })

    return results_list


def stats_dice(results_list):
    """
    Compute summary statistics over per-patient Dice results.
    """
    dice_before_all = np.array([r["dice_before"] for r in results_list], dtype=float)
    dice_after_all = np.array([r["dice_after"] for r in results_list], dtype=float)
    dice_gain_all = np.array([r["dice_gain"] for r in results_list], dtype=float)

    stats = {
        "n_patients": len(results_list),
        "mean_dice_before": float(np.mean(dice_before_all)),
        "mean_dice_after": float(np.mean(dice_after_all)),
        "mean_dice_gain": float(np.mean(dice_gain_all)),
        "median_dice_before": float(np.median(dice_before_all)),
        "median_dice_after": float(np.median(dice_after_all)),
        "median_dice_gain": float(np.median(dice_gain_all)),
        "min_dice_gain": float(np.min(dice_gain_all)),
        "max_dice_gain": float(np.max(dice_gain_all)),
        "n_improved": int(np.sum(dice_gain_all > 0)),
        "n_equal": int(np.sum(np.isclose(dice_gain_all, 0))),
        "n_worse": int(np.sum(dice_gain_all < 0)),
    }

    return stats