# src/registration_functions.py
"""
Functions defined for the image registration
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import glob

import importdata_functions as idf
import resampling_functions as rsf
from paths import * 

def mask_centroid_bbox(mask: np.ndarray, absencevalue=0):
    """
    Compute centroid + bbox from  mask.

    Returns
    -------
    centroid : tuple[float, float, float]
        Centroid in voxel coordinates (x, y, z), as floats.
    bbox : tuple[int, int, int, int, int, int]
        Bounding box (xmin, xmax, ymin, ymax, zmin, zmax) with xmax/ymax/zmax
        as *exclusive* indices (Python slicing-friendly).
    roi_nvox : int
        Number of ROI voxels.
    """
    coords = np.argwhere(mask != absencevalue)

    # centroid in voxel coordinates
    centroid = tuple(coords.mean(axis=0).astype(float))  # (x, y, z)

    # bbox, python-slice friendly: [xmin:xmax, ymin:ymax, zmin:zmax]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # make max exclusive

    xmin, ymin, zmin = (int(mins[0]), int(mins[1]), int(mins[2]))
    xmax, ymax, zmax = (int(maxs[0]), int(maxs[1]), int(maxs[2]))

    bbox = (xmin, xmax, ymin, ymax, zmin, zmax)

    # Number of voxels 
    roi_nvox = int(coords.shape[0])

    return centroid, bbox, roi_nvox


def roi_stats_from_mask(mask: np.ndarray, absencevalue=0, spacing_xyz=None):
    """
    Returns centroid, bbox, roi voxel count, and comparisons to the full mask.

    spacing_xyz : (sx, sy, sz) in mm. If provided, also returns:
        - centroid_mm
        - roi_volume_mm3
    """
    centroid, bbox, nvox = mask_centroid_bbox(mask, absencevalue=absencevalue)

    X, Y, Z = mask.shape

    total_nvox = int(mask.size)
    roi_fraction = float(nvox / total_nvox)

    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    out = {
        "centroid_vox": centroid,
        "bbox": bbox,  # ROI bbox

        # full mask bounds
        "full_bounds": (0, X-1, 0, Y-1, 0, Z-1),

        # voxel statistics
        "roi_nvox": nvox,
        "total_nvox": total_nvox,
        "roi_fraction": roi_fraction,
    }

    if spacing_xyz is not None:
        sx, sy, sz = spacing_xyz
        out["roi_volume_mm3"] = float(nvox * sx * sy * sz)

        out["centroid_mm"] = (
            centroid[0] * sx,
            centroid[1] * sy,
            centroid[2] * sz
        )

    return out

def mask_allbboxsizes(onlytraining=False, limit = 1000,  top_percent=95):
    """
    """
    # Load all resampled masks
    all_gt_res = idf.load_allgt_res(onlytraining=onlytraining)
    # Get bboxes
    bboxinfo_list = []
    # centroid_list = []
    for i in range(min(len(all_gt_res), limit)):
        #Get mask bbox
        mask_path = all_gt_res[i]
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(np.uint8)  # (X,Y,Z)
        centroid, bbox, roi_nvox = mask_centroid_bbox(mask)
        # centroid_list.append(centroid)
        # Calculations 
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        bboxinfo_list.append({
            "path": mask_path,
            "bbox": bbox,
            "bbox_size": (dx, dy, dz),
            "centroid": centroid
            # "roi_nvox": roi_nvox
        })

    # Convert to array for statistics
    bboxsize_array = np.array([entry["bbox_size"] for entry in bboxinfo_list])

    stats = {
        "n_masks": len(bboxinfo_list),

        "max_x": int(np.max(bboxsize_array[:, 0])),
        "max_y": int(np.max(bboxsize_array[:, 1])),
        "max_z": int(np.max(bboxsize_array[:, 2])),

        f"p{top_percent}_x": float(np.percentile(bboxsize_array[:, 0], top_percent)),
        f"p{top_percent}_y": float(np.percentile(bboxsize_array[:, 1], top_percent)),
        f"p{top_percent}_z": float(np.percentile(bboxsize_array[:, 2], top_percent)),

        "mean_x": float(np.mean(bboxsize_array[:, 0])),
        "mean_y": float(np.mean(bboxsize_array[:, 1])),
        "mean_z": float(np.mean(bboxsize_array[:, 2])),

        "median_x": float(np.median(bboxsize_array[:, 0])),
        "median_y": float(np.median(bboxsize_array[:, 1])),
        "median_z": float(np.median(bboxsize_array[:, 2])),
    }

    return bboxinfo_list, stats

import numpy as np

def crop_pad_around_centroid(img_nii, mask_nii, centroid, crop_shape=(128, 128, 32)):
    """
    Crop a fixed-size box around a centroid and return NIfTI images.

    Parameters
    ----------
    img_nii : nib.Nifti1Image
    mask_nii : nib.Nifti1Image
    centroid : (x,y,z) voxel coordinates
    crop_shape : (crop_x, crop_y, crop_z)

    Returns
    -------
    img_crop_nii
    mask_crop_nii
    """

    image = img_nii.get_fdata().astype(np.float32)
    mask = mask_nii.get_fdata().astype(np.uint8)
    X, Y, Z = image.shape
    crop_x, crop_y, crop_z = crop_shape

    # Convert centroid to nearest voxel index
    cx, cy, cz = [int(round(c)) for c in centroid]

    # Desired crop bounds in the original image
    x0 = cx - crop_x // 2
    x1 = x0 + crop_x
    y0 = cy - crop_y // 2
    y1 = y0 + crop_y
    z0 = cz - crop_z // 2
    z1 = z0 + crop_z

    # Clip crop bounds to image limits
    x0_img = max(0, x0)
    x1_img = min(X, x1)
    y0_img = max(0, y0)
    y1_img = min(Y, y1)
    z0_img = max(0, z0)
    z1_img = min(Z, z1)

    # Corresponding bounds in the output cropped image
    x0_out = x0_img - x0
    x1_out = x0_out + (x1_img - x0_img)
    y0_out = y0_img - y0
    y1_out = y0_out + (y1_img - y0_img)
    z0_out = z0_img - z0
    z1_out = z0_out + (z1_img - z0_img)

    # Create output arrays
    image_crop = np.zeros(crop_shape, dtype=image.dtype)
    mask_crop = np.zeros(crop_shape, dtype=mask.dtype)
    image_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = image[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]
    mask_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = mask[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]

    # --- update affine ---
    affine = img_nii.affine.copy()
    voxel_shift = np.array([x0, y0, z0])
    world_shift = affine[:3, :3] @ voxel_shift
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + world_shift

    img_crop_nii = nib.Nifti1Image(image_crop, new_affine)
    mask_crop_nii = nib.Nifti1Image(mask_crop, new_affine)

    return img_crop_nii, mask_crop_nii


def crop_all_frames(crop_shape=(128,128,32), limit=1000):
    """
    """
    # Load resampled frames and masks paths
    all_img_res, all_gt_res = idf.load_allframes_resampled()

    for i in range(min(len(all_img_res), limit)):

        # Load image and mask
        img_path = all_img_res[i]
        mask_path = img_path.replace("_resampled.nii.gz", "_resampled_gt.nii.gz")
        # mask_path = all_gt_res[i]
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        # centroid from mask
        mask = mask_nii.get_fdata().astype(np.uint8)
        centroid, bbox, roi_nvox = mask_centroid_bbox(mask)

        # crop
        if img_nii.shape != mask_nii.shape:
            print("Shape mismatch:")
            print("img :", img_path, img_nii.shape)
            print("mask:", mask_path, mask_nii.shape)
            raise ValueError("Image and mask shapes differ before cropping.")

        img_crop_nii, mask_crop_nii = crop_pad_around_centroid(
            img_nii,
            mask_nii,
            centroid,
            crop_shape=crop_shape
        )

        # build output names and save
        path = Path(img_path)
        filename = path.name
        patient_id = filename.split("_")[0]
        frame_id = filename.split("_")[1]
        out_folder = path_tempodata_folder + "cropped_frames/"
        img_out = out_folder  + patient_id + "_" + frame_id + "_cropped.nii.gz"
        mask_out = out_folder  + patient_id + "_" + frame_id + "_cropped_gt.nii.gz"
        nib.save(img_crop_nii, img_out)
        nib.save(mask_crop_nii, mask_out)

#########################################


def _sitk_to_nib(img_sitk, reference_nib, dtype=np.float32):
    """
    Convert SimpleITK image back to nibabel NIfTI, using the reference affine.
    """
    arr_zyx = sitk.GetArrayFromImage(img_sitk)      # (Z,Y,X)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0)).astype(dtype)  # back to (X,Y,Z)

    nii = nib.Nifti1Image(arr_xyz, reference_nib.affine)
    nii.header.set_zooms(reference_nib.header.get_zooms()[:3])

    return nii


def rigid_register_one_patient(fixed_img_nii, fixed_mask_nii, 
                                                                      moving_img_nii, moving_mask_nii, 
                                                                    #   original_img_nii, original_mask_nii, 
                                                                      number_of_iterations=200):
    """
    Rigidly register one moving patient to one fixed reference.

    Registration is driven by binary masks.
    The resulting rigid transform is then applied to:
      - moving image  (linear interpolation)
      - moving mask   (nearest-neighbor interpolation)

    Returns
    -------
    registered_img_nii : nib.Nifti1Image
    registered_mask_nii : nib.Nifti1Image
    final_transform : sitk.Transform
    """

    # Convert all necessary images from nii to sitk
    fixed_img_sitk = rsf._make_sitk_from_nib(fixed_img_nii)
    moving_img_sitk = rsf._make_sitk_from_nib(moving_img_nii)
     # Convert original masks (multi-label)
    fixed_mask_sitk = rsf._make_sitk_from_nib(fixed_mask_nii)
    moving_mask_sitk = rsf._make_sitk_from_nib(moving_mask_nii)
    # Create binary masks for registration
    fixed_mask_bin_sitk = sitk.Cast(fixed_mask_sitk > 0, sitk.sitkFloat32)
    moving_mask_bin_sitk = sitk.Cast(moving_mask_sitk > 0, sitk.sitkFloat32)

        # Initialize rigid transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_mask_bin_sitk,
        moving_mask_bin_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Registration setup
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

    # Run registration
    final_transform = registration_method.Execute(fixed_mask_bin_sitk, moving_mask_bin_sitk)

    # Apply transform to image
    registered_img_sitk = sitk.Resample(moving_img_sitk, fixed_img_sitk, 
                                                                                  final_transform, sitk.sitkLinear, 0.0, moving_img_sitk.GetPixelID())

    # Apply transform to original multi-label mask with nearest neighbor
    registered_mask_sitk = sitk.Resample(moving_mask_sitk, fixed_mask_sitk,
                                                                                     final_transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # Convert back to nibabel
    registered_img_nii = _sitk_to_nib(registered_img_sitk, fixed_img_nii, dtype=np.float32)
    registered_mask_nii = _sitk_to_nib(registered_mask_sitk, fixed_mask_nii, dtype=np.uint8)

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

def register_all_frames(reference_patient="patient001", limit=1000):
    """
    Register all cropped frames to a fixed reference patient and save results.
    The frame number is taken dynamically from filenames (frame01, frame04, etc.).
    """
    # Load all paths from cropped frames (128x128x32 default)
    all_img_crop = sorted(idf.load_allcroppedframes())
    # Out folder 
    out_folder = Path(path_tempodata_folder) / "registered_frames"
    out_folder.mkdir(parents=True, exist_ok=True)

    # --- find reference patient and guess his name and frame number
    ref_img_path = None
    for p in all_img_crop:
        if reference_patient in Path(p).name:
            ref_img_path = p
            break
    if ref_img_path is None:
        raise ValueError(f"Reference patient {reference_patient} not found.")
    ref_path = Path(ref_img_path)
    ref_patient_id, ref_frame_id = ref_path.stem.split("_")[:2]
    # Load ref patient
    fixed_img_nii = nib.load(ref_img_path)
    ref_mask_path = ref_img_path.replace("_cropped.nii.gz", "_cropped_gt.nii.gz")
    fixed_mask_nii = nib.load(ref_mask_path)

    # Registration
    for i in range(min(len(all_img_crop), limit)):
        # Guess patient name and frame number
        img_path = all_img_crop[i]
        path = Path(img_path)
        patient_id, frame_id = path.stem.split("_")[:2]
        # Make paths to save from this 2 infos
        img_out = out_folder / f"{patient_id}_{frame_id}_registered.nii.gz"
        mask_out = out_folder / f"{patient_id}_{frame_id}_registered_gt.nii.gz"
        # Skip reference patient
        if patient_id == ref_patient_id and frame_id == ref_frame_id:
            nib.save(fixed_img_nii, img_out)
            nib.save(fixed_mask_nii, mask_out)
            # print(f"[{i+1}/{len(all_img_crop)}] Saved reference {img_out.name}")
            continue
        # Do registration and save
        mask_path = str(path).replace("_cropped.nii.gz", "_cropped_gt.nii.gz")
        moving_img_nii = nib.load(img_path)
        moving_mask_nii = nib.load(mask_path)
        reg_img_nii, reg_mask_nii, final_transform = rigid_register_one_patient(
            fixed_img_nii,
            fixed_mask_nii,
            moving_img_nii,
            moving_mask_nii
        )
        nib.save(reg_img_nii, img_out)
        nib.save(reg_mask_nii, mask_out)
        print(f"[{i+1}/{min(len(all_img_crop), limit)}] Saved {img_out.name}")

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

def dice_all_patients(reference_patient="patient001", cropped_folder="cropped_frames", registered_folder="registered_frames"):
    """
    Compute Dice before/after registration for all patients.

    Dice before:
        cropped moving mask vs cropped reference mask
    Dice after:
        registered moving mask vs cropped reference mask

    Returns
    -------
    results_list : list of dict
        Per-patient dice results
    """

    # --- load all cropped masks ---
    cropped_mask_paths = sorted(glob.glob(path_tempodata_folder + cropped_folder + "/patient*_frame*_cropped_gt.nii.gz"))
    # --- load all registered masks ---
    registered_mask_paths = sorted(glob.glob(path_tempodata_folder + registered_folder + "/patient*_frame*_registered_gt.nii.gz"))

    # --- find reference mask dynamically in cropped folder ---
    ref_mask_path = None
    for p in cropped_mask_paths:
        if Path(p).name.startswith(reference_patient + "_"):
            ref_mask_path = p
            break
    if ref_mask_path is None:
        raise ValueError(f"Reference patient {reference_patient} not found in {cropped_folder}.")
    ref_mask = nib.load(ref_mask_path).get_fdata()

    # --- keep same number of files ---
    n_files = min(len(cropped_mask_paths), len(registered_mask_paths))
    if len(cropped_mask_paths) != len(registered_mask_paths):
        print(f"Warning: different number of files.")
        print(f"cropped masks   : {len(cropped_mask_paths)}")
        print(f"registered masks: {len(registered_mask_paths)}")
        print(f"Using first {n_files} paired files after sorting.")

    # --- loop over paired masks ---
    results_list = []
    for i in range(n_files):
        cropped_mask_path = cropped_mask_paths[i]
        reg_mask_path = registered_mask_paths[i]
        cropped_name = Path(cropped_mask_path).name.replace("_cropped_gt.nii.gz", "")
        reg_name = Path(reg_mask_path).name.replace("_registered_gt.nii.gz", "")
        cropped_patient_id, cropped_frame_id = cropped_name.split("_")[:2]
        reg_patient_id, reg_frame_id = reg_name.split("_")[:2]
        # Minimal consistency check
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
        # Dice before/after against reference
        dice_before = dice_score(ref_mask, cropped_mask)
        dice_after = dice_score(ref_mask, reg_mask)
        # Store results
        results_list.append({
            "patient_id": patient_id,
            "frame_id": frame_id,
            # "cropped_mask_path": cropped_mask_path,
            # "registered_mask_path": reg_mask_path,
            "dice_before": float(dice_before),
            "dice_after": float(dice_after),
            "dice_gain": float(dice_after - dice_before),
        })

    return results_list

def stats_dice(results_list):
    """
    """
    # Summary stats
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