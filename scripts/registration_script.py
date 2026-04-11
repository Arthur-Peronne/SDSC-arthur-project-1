# scripts/registration_script.py
"""
Script to perform the image resampling and the following registration
"""

import nibabel as nib
import numpy as np

from src.config import TEMPODATA_FOLDER
from src.data import registration as rgt
from src.data import geometry as rgg
from src.data import resampling as rsp
from src.visualization import mri_plots as mrp


# FIRST TESTS -> OK

# patient1 = "149"
# gt = "_gt" # "_gt" or ""
# target_spacing = [1.5, 1.5, 3.15]
# img_path = TEMPODATA_FOLDER / "resampled_frames" / f"patient{patient1}_frame01_resampled.nii.gz"
# mask_path = TEMPODATA_FOLDER / "resampled_frames" / f"patient{patient1}_frame01_resampled_gt.nii.gz"

# # RESAMPLING all
# rsp.resample_all(target_spacing)

# # CROPPING
# img_nii = nib.load(img_path)
# mask_nii = nib.load(mask_path)
# mask = mask_nii.get_fdata().astype(np.uint8)
# centroid, bbox, roi_nvox = rgg.mask_centroid_bbox(mask)
# image_crop, mask_crop = rgg.crop_pad_around_centroid(img_nii, mask_nii, centroid=centroid)
# # Plot cropped images
# img_path_cropped = TEMPODATA_FOLDER / "cropped_frames" / f"patient{patient1}_frame01_cropped.nii.gz"
# mask_path_cropped = TEMPODATA_FOLDER / "cropped_frames" / f"patient{patient1}_frame01_cropped_gt.nii.gz"
# image_crop = nib.load(img_path_cropped)
# mask_crop = nib.load(mask_path_cropped)
# mrp.plot_oneimg(image_crop, patient_str=patient1, file_str="frame01", details_str="CROPPED")
# mrp.plot_onemask(mask_crop, patient_str=patient1, file_str="frame01_gt", details_str="CROPPED")
# mrp.plot_oneimagemask(image_crop, mask_crop, patient_str=patient1, file_str="frame01", details_str="CROPPED")
# # Crop all
# rgg.crop_all_frames()

# REGISTRATION
# ref = "patient001"
# mov = "patient003"
# fixed_img_full_nii = nib.load(TEMPODATA_FOLDER / "resampled_frames" / f"{ref}_frame01_resampled.nii.gz")
# fixed_mask_full_nii = nib.load(TEMPODATA_FOLDER / "resampled_frames" / f"{ref}_frame01_resampled_gt.nii.gz")
# fixed_img_crop_nii = nib.load(TEMPODATA_FOLDER / "cropped_frames" / f"{ref}_frame01_cropped.nii.gz")
# fixed_mask_crop_nii = nib.load(TEMPODATA_FOLDER / "cropped_frames" / f"{ref}_frame01_cropped_gt.nii.gz")
# moving_img_full_nii = nib.load(TEMPODATA_FOLDER / "resampled_frames" / f"{mov}_frame01_resampled.nii.gz")
# moving_mask_full_nii = nib.load(TEMPODATA_FOLDER / "resampled_frames" / f"{mov}_frame01_resampled_gt.nii.gz")
# moving_img_crop_nii = nib.load(TEMPODATA_FOLDER / "cropped_frames" / f"{mov}_frame01_cropped.nii.gz")
# moving_mask_crop_nii = nib.load(TEMPODATA_FOLDER / "cropped_frames" / f"{mov}_frame01_cropped_gt.nii.gz")
# reg_img_full_nii, reg_mask_full_nii, T = rgt.rigid_register_one_patient(fixed_img_full_nii, fixed_mask_full_nii, fixed_mask_crop_nii, moving_mask_crop_nii, moving_img_full_nii, moving_mask_full_nii)
# reg_img_nii, reg_mask_nii = rgg.crop_to_reference_window(reg_img_full_nii, reg_mask_full_nii, fixed_img_crop_nii)
# rgt.print_rigid_transform_info(T)
# # Plot registration and compare
# mrp.plot_oneimg(fixed_img_full_nii, patient_str=ref, file_str="IMG", details_str="referenceFULL")
# mrp.plot_onemask(fixed_mask_full_nii, patient_str=ref, file_str="MASK", details_str="referenceFULL")
# mrp.plot_oneimagemask(fixed_img_full_nii, fixed_mask_full_nii, patient_str=ref, file_str="SUP", details_str="referenceFULL")
# mrp.plot_oneimg(fixed_img_crop_nii, patient_str=ref, file_str="IMG", details_str="referenceCROP")
# mrp.plot_onemask(fixed_mask_crop_nii, patient_str=ref, file_str="MASK", details_str="referenceCROP")
# mrp.plot_oneimagemask(fixed_img_crop_nii, fixed_mask_crop_nii, patient_str=ref, file_str="SUP", details_str="referenceCROP")
# mrp.plot_oneimg(moving_img_full_nii, patient_str=mov, file_str="IMG", details_str="originalFULL")
# mrp.plot_onemask(moving_mask_full_nii, patient_str=mov, file_str="MASK", details_str="originalFULL")
# mrp.plot_oneimagemask(moving_img_full_nii, moving_mask_full_nii, patient_str=mov, file_str="SUP", details_str="originalFULL")
# mrp.plot_oneimg(moving_img_crop_nii, patient_str=mov, file_str="IMG", details_str="originalCROP")
# mrp.plot_onemask(moving_mask_crop_nii, patient_str=mov, file_str="MASK", details_str="originalCROP")
# mrp.plot_oneimagemask(moving_img_crop_nii, moving_mask_crop_nii, patient_str=mov, file_str="SUP", details_str="originalCROP")
# mrp.plot_oneimg(reg_img_nii, patient_str=mov, file_str="IMG", details_str="registered")
# mrp.plot_onemask(reg_mask_nii, patient_str=mov, file_str="MASK", details_str="registered")
# mrp.plot_oneimagemask(reg_img_nii, reg_mask_nii, patient_str=mov, file_str="SUP", details_str="registered")
# mrp.plot_oneimagemask(fixed_img_crop_nii, reg_mask_nii, patient_str=f"{mov}mixed", file_str="SUP", details_str="regmask_refimg")
# mrp.plot_oneimagemask(fixed_mask_crop_nii, reg_mask_nii, patient_str=f"{mov}mixed", file_str="SUP", details_str="regmask_refmask")
# # Registration for all
# rgt.register_all_frames()

# DICE
# results_list = rgt.dice_all_patients(registered_folder="registered_framesBIS")
# stats = rgt.stats_dice(results_list)
# print(results_list, stats)