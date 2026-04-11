# src/data/geometry.py
"""
Geometry utilities for cardiac MRI preprocessing:
centroid computation, bounding boxes, cropping and padding.
No dependency on SimpleITK.
"""

import numpy as np
import nibabel as nib
from pathlib import Path

from src.config import TEMPODATA_FOLDER
from src.data import importdata as ipd


def mask_centroid_bbox(mask: np.ndarray, absencevalue=0):
    """
    Compute centroid and bounding box from a mask.

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

    centroid = tuple(coords.mean(axis=0).astype(float))  # (x, y, z)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # make max exclusive

    xmin, ymin, zmin = (int(mins[0]), int(mins[1]), int(mins[2]))
    xmax, ymax, zmax = (int(maxs[0]), int(maxs[1]), int(maxs[2]))

    bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
    roi_nvox = int(coords.shape[0])

    return centroid, bbox, roi_nvox


def roi_stats_from_mask(mask: np.ndarray, absencevalue=0, spacing_xyz=None):
    """
    Return centroid, bbox, roi voxel count, and comparisons to the full mask.

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
        "bbox": bbox,
        "full_bounds": (0, X-1, 0, Y-1, 0, Z-1),
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


def mask_allbboxsizes(onlytraining=False, limit=1000, top_percent=95):
    """
    Compute bounding box sizes for all resampled masks.

    Returns
    -------
    bboxinfo_list : list of dict
    stats : dict
        Summary statistics (max, percentile, mean, median) per axis.
    """
    all_gt_res = ipd.load_allgt_res(onlytraining=onlytraining)

    bboxinfo_list = []
    for i in range(min(len(all_gt_res), limit)):
        mask_path = all_gt_res[i]
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(np.uint8)
        centroid, bbox, roi_nvox = mask_centroid_bbox(mask)

        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        bboxinfo_list.append({
            "path": mask_path,
            "bbox": bbox,
            "bbox_size": (dx, dy, dz),
            "centroid": centroid
        })

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


def crop_pad_around_centroid(img_nii, mask_nii, centroid, crop_shape=(128, 128, 32)):
    """
    Crop a fixed-size box around a centroid and return NIfTI images.

    Parameters
    ----------
    img_nii : nib.Nifti1Image
    mask_nii : nib.Nifti1Image
    centroid : (x, y, z) voxel coordinates
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

    cx, cy, cz = [int(round(c)) for c in centroid]

    x0 = cx - crop_x // 2
    x1 = x0 + crop_x
    y0 = cy - crop_y // 2
    y1 = y0 + crop_y
    z0 = cz - crop_z // 2
    z1 = z0 + crop_z

    x0_img = max(0, x0)
    x1_img = min(X, x1)
    y0_img = max(0, y0)
    y1_img = min(Y, y1)
    z0_img = max(0, z0)
    z1_img = min(Z, z1)

    x0_out = x0_img - x0
    x1_out = x0_out + (x1_img - x0_img)
    y0_out = y0_img - y0
    y1_out = y0_out + (y1_img - y0_img)
    z0_out = z0_img - z0
    z1_out = z0_out + (z1_img - z0_img)

    image_crop = np.zeros(crop_shape, dtype=image.dtype)
    mask_crop = np.zeros(crop_shape, dtype=mask.dtype)
    image_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = image[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]
    mask_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = mask[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]

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

    x0 = int(round((ref_affine[0, 3] - full_affine[0, 3]) / full_affine[0, 0]))
    y0 = int(round((ref_affine[1, 3] - full_affine[1, 3]) / full_affine[1, 1]))
    z0 = int(round((ref_affine[2, 3] - full_affine[2, 3]) / full_affine[2, 2]))

    x1 = x0 + crop_x
    y1 = y0 + crop_y
    z1 = z0 + crop_z

    x0_img = max(0, x0)
    x1_img = min(X, x1)
    y0_img = max(0, y0)
    y1_img = min(Y, y1)
    z0_img = max(0, z0)
    z1_img = min(Z, z1)

    x0_out = x0_img - x0
    x1_out = x0_out + (x1_img - x0_img)
    y0_out = y0_img - y0
    y1_out = y0_out + (y1_img - y0_img)
    z0_out = z0_img - z0
    z1_out = z0_out + (z1_img - z0_img)

    img_crop = np.zeros((crop_x, crop_y, crop_z), dtype=image.dtype)
    mask_crop = np.zeros((crop_x, crop_y, crop_z), dtype=mask.dtype)

    img_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = image[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]
    mask_crop[x0_out:x1_out, y0_out:y1_out, z0_out:z1_out] = mask[x0_img:x1_img, y0_img:y1_img, z0_img:z1_img]

    img_crop_nii = nib.Nifti1Image(img_crop, ref_affine)
    mask_crop_nii = nib.Nifti1Image(mask_crop, ref_affine)

    return img_crop_nii, mask_crop_nii


def crop_all_frames(crop_shape=(128, 128, 32), limit=1000):
    """
    Crop all resampled frames around their cardiac mask centroid and save results.
    """
    all_img_res, all_gt_res = ipd.load_allframes_resampled()

    for i in range(min(len(all_img_res), limit)):

        img_path = all_img_res[i]
        mask_path = img_path.replace("_resampled.nii.gz", "_resampled_gt.nii.gz")
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        mask = mask_nii.get_fdata().astype(np.uint8)
        centroid, bbox, roi_nvox = mask_centroid_bbox(mask)

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

        path = Path(img_path)
        filename = path.name
        patient_id = filename.split("_")[0]
        frame_id = filename.split("_")[1]
        out_folder = TEMPODATA_FOLDER / "cropped_frames"
        img_out = out_folder / f"{patient_id}_{frame_id}_cropped.nii.gz"
        mask_out = out_folder / f"{patient_id}_{frame_id}_cropped_gt.nii.gz"
        nib.save(img_crop_nii, img_out)
        nib.save(mask_crop_nii, mask_out)