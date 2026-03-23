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

def crop_to_reference_window(img_nii, mask_nii, reference_img_crop_nii):
    """
    Crop a full registered image/mask to the exact spatial window of the fixed reference crop.
    If the reference window extends outside the full image, pad with zeros.

    Parameters
    ----------
    img_nii : nib.Nifti1Image
        Full registered image in the full fixed space.
    mask_nii : nib.Nifti1Image
        Full registered mask in the full fixed space.
    reference_img_crop_nii : nib.Nifti1Image
        Cropped fixed reference image defining the output window.

    Returns
    -------
    img_crop_nii
    mask_crop_nii
    """
    image = img_nii.get_fdata().astype(np.float32)
    mask = mask_nii.get_fdata().astype(np.uint8)

    X, Y, Z = image.shape

    ref_affine = reference_img_crop_nii.affine
    ref_shape = reference_img_crop_nii.shape
    crop_x, crop_y, crop_z = ref_shape

    full_affine = img_nii.affine

    # Recover the crop start indices of the reference window in the full fixed space
    x0 = int(round((ref_affine[0, 3] - full_affine[0, 3]) / full_affine[0, 0]))
    y0 = int(round((ref_affine[1, 3] - full_affine[1, 3]) / full_affine[1, 1]))
    z0 = int(round((ref_affine[2, 3] - full_affine[2, 3]) / full_affine[2, 2]))

    x1 = x0 + crop_x
    y1 = y0 + crop_y
    z1 = z0 + crop_z

    # Clip to valid image bounds
    x0_img = max(0, x0)
    x1_img = min(X, x1)
    y0_img = max(0, y0)
    y1_img = min(Y, y1)
    z0_img = max(0, z0)
    z1_img = min(Z, z1)

    # Corresponding bounds in output crop
    x0_out = x0_img - x0
    x1_out = x0_out + (x1_img - x0_img)
    y0_out = y0_img - y0
    y1_out = y0_out + (y1_img - y0_img)
    z0_out = z0_img - z0
    z1_out = z0_out + (z1_img - z0_img)

    # Allocate fixed-size outputs
    img_crop = np.zeros((crop_x, crop_y, crop_z), dtype=image.dtype)
    mask_crop = np.zeros((crop_x, crop_y, crop_z), dtype=mask.dtype)

    # Copy valid overlap
    img_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = image[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]
    mask_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = mask[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]

    img_crop_nii = nib.Nifti1Image(img_crop, ref_affine)
    mask_crop_nii = nib.Nifti1Image(mask_crop, ref_affine)

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


def rigid_register_one_patient(fixed_img_full_nii, fixed_mask_full_nii, 
                                                                    fixed_mask_crop_nii, moving_mask_crop_nii,
                                                                    moving_img_full_nii, moving_mask_full_nii,
                                                                    number_of_iterations=200
                                                                    ):
    """
    Estimate rigid transform on cropped binary masks,
    then apply it to full resampled image and full multi-label mask,
    resampling onto the full fixed grid.
    """

    # --- SITK conversion ---
    fixed_img_full_sitk = rsf._make_sitk_from_nib(fixed_img_full_nii)
    fixed_mask_full_sitk = rsf._make_sitk_from_nib(fixed_mask_full_nii)

    fixed_mask_crop_sitk = rsf._make_sitk_from_nib(fixed_mask_crop_nii)
    moving_mask_crop_sitk = rsf._make_sitk_from_nib(moving_mask_crop_nii)

    moving_img_full_sitk = rsf._make_sitk_from_nib(moving_img_full_nii)
    moving_mask_full_sitk = rsf._make_sitk_from_nib(moving_mask_full_nii)

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
    registered_img_nii = _sitk_to_nib(
        registered_img_sitk,
        fixed_img_full_nii,
        dtype=np.float32
    )
    registered_mask_nii = _sitk_to_nib(
        registered_mask_sitk,
        fixed_mask_full_nii,
        dtype=np.uint8
    )

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

    # Load all cropped image paths
    all_img_crop = sorted(idf.load_allcroppedframes())

    # Output folder
    out_folder = Path(path_tempodata_folder) / "registered_framesBIS"
    out_folder.mkdir(parents=True, exist_ok=True)

    # --- find reference cropped image dynamically ---
    ref_img_crop_path = None
    for p in all_img_crop:
        if Path(p).name.startswith(reference_patient + "_"):
            ref_img_crop_path = p
            break
    if ref_img_crop_path is None:
        raise ValueError(f"Reference patient {reference_patient} not found.")

    ref_crop_path = Path(ref_img_crop_path)
    ref_patient_id, ref_frame_id = ref_crop_path.stem.split("_")[:2]

    # Load fixed cropped image/mask
    fixed_img_crop_nii = nib.load(ref_img_crop_path)
    fixed_mask_crop_path = ref_img_crop_path.replace("_cropped.nii.gz", "_cropped_gt.nii.gz")
    fixed_mask_crop_nii = nib.load(fixed_mask_crop_path)

    # Load fixed full resampled image/mask
    fixed_img_full_path = (
        path_tempodata_folder + "resampled_frames/" + ref_patient_id + "_" + ref_frame_id + "_resampled.nii.gz"
    )
    fixed_mask_full_path = (
        path_tempodata_folder + "resampled_frames/" + ref_patient_id + "_" + ref_frame_id + "_resampled_gt.nii.gz"
    )
    fixed_img_full_nii = nib.load(fixed_img_full_path)
    fixed_mask_full_nii = nib.load(fixed_mask_full_path)

    # Optional safety check
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

        # Moving cropped mask for transform estimation
        moving_mask_crop_path = str(path).replace("_cropped.nii.gz", "_cropped_gt.nii.gz")
        moving_mask_crop_nii = nib.load(moving_mask_crop_path)

        # Moving full resampled image/mask for transform application
        moving_img_full_path = (
            path_tempodata_folder + "resampled_frames/" + patient_id + "_" + frame_id + "_resampled.nii.gz"
        )
        moving_mask_full_path = (
            path_tempodata_folder + "resampled_frames/" + patient_id + "_" + frame_id + "_resampled_gt.nii.gz"
        )

        moving_img_full_nii = nib.load(moving_img_full_path)
        moving_mask_full_nii = nib.load(moving_mask_full_path)

        # Registration in full fixed space
        reg_img_full_nii, reg_mask_full_nii, final_transform = rigid_register_one_patient(
            fixed_img_full_nii, fixed_mask_full_nii, fixed_mask_crop_nii,
            moving_mask_crop_nii,
            moving_img_full_nii, moving_mask_full_nii
        )

        # Optional crop after registration
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
        # print(cropped_mask.shape, reg_mask.shape)
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