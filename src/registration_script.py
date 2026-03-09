# src/registration_script.py
"""
Script to perform the image resampling and the following registration
"""

import nibabel as nib
import numpy as np 

from paths import *
import registration_functions as rgf
import resampling_functions as rsf 
import importdata_functions as idf
import various_plots as vpl
import visualizeMRI_functions as vmf

# FIRST TESTS -> OK 

# patient1 = "149"
# gt = "_gt" # "_gt" or ""
# target_spacing = [1.5, 1.5, 3.15]
# img_path = path_tempodata_folder + "resampled_frames/patient" + patient1 + "_frame01_resampled.nii.gz"
# mask_path = path_tempodata_folder + "resampled_frames/patient" + patient1 + "_frame01_resampled_gt.nii.gz"

# # RESAMPLING all 
# rsf.resample_all(target_spacing)

# # CROPPING
# img_nii = nib.load(img_path)
# mask_nii = nib.load(mask_path)
# mask = mask_nii.get_fdata().astype(np.uint8)
# centroid, bbox, roi_nvox = rgf.mask_centroid_bbox(mask)
# image_crop, mask_crop = rgf.crop_pad_around_centroid(img_nii, mask_nii, centroid=centroid)
# # Plot cropped images 
# img_path_cropped = path_tempodata_folder + "cropped_frames/patient" + patient1 + "_frame01_cropped.nii.gz"
# mask_path_copped = path_tempodata_folder + "cropped_frames/patient" + patient1 + "_frame01_cropped_gt.nii.gz"
# image_crop = nib.load(img_path_cropped)
# mask_crop = nib.load(mask_path_copped)
# vmf.plot_oneimg(image_crop, patient_str =patient1, file_str ="frame01", details_str="CROPPED")
# vmf.plot_onemask(mask_crop, patient_str =patient1, file_str ="frame01"+"_gt", details_str="CROPPED")
# vmf.plot_oneimagemask(image_crop, mask_crop, patient_str =patient1, file_str ="frame01", details_str="CROPPED")
#  Crop all
# rgf.crop_all_frames()

# REGISTRATION 
# ref = "patient001"
# mov = "patient022"
# fixed_img_nii = nib.load(path_tempodata_folder + "cropped_frames/" + ref + "_frame01_cropped.nii.gz")
# fixed_mask_nii = nib.load(path_tempodata_folder + "cropped_frames/" + ref + "_frame01_cropped_gt.nii.gz")
# moving_img_nii = nib.load(path_tempodata_folder + "cropped_frames/" + mov + "_frame01_cropped.nii.gz")
# moving_mask_nii = nib.load(path_tempodata_folder + "cropped_frames/" + mov + "_frame01_cropped_gt.nii.gz")
# reg_img_nii, reg_mask_nii, T = rgf.rigid_register_one_patient(fixed_img_nii, fixed_mask_nii, moving_img_nii, moving_mask_nii)
# rgf.print_rigid_transform_info(T)
# # Plot registration and compare
# vmf.plot_oneimg(fixed_img_nii, patient_str =ref, file_str ="IMG", details_str="reference")
# vmf.plot_onemask(fixed_mask_nii, patient_str =ref, file_str ="MASK", details_str="reference")
# vmf.plot_oneimagemask(fixed_img_nii, fixed_mask_nii, patient_str =ref, file_str ="SUP", details_str="reference")
# vmf.plot_oneimg(moving_img_nii, patient_str =mov, file_str ="IMG", details_str="original")
# vmf.plot_onemask(moving_mask_nii, patient_str =mov, file_str ="MASK", details_str="original")
# vmf.plot_oneimagemask(moving_img_nii, moving_mask_nii, patient_str =mov, file_str ="SUP", details_str="original")
# vmf.plot_oneimg(reg_img_nii, patient_str =mov, file_str ="IMG", details_str="registered")
# vmf.plot_onemask(reg_mask_nii, patient_str =mov, file_str ="MASK", details_str="registered")
# vmf.plot_oneimagemask(reg_img_nii, reg_mask_nii, patient_str =mov, file_str ="SUP", details_str="registered")
# vmf.plot_oneimagemask(fixed_img_nii, reg_mask_nii, patient_str = mov+"mixed", file_str ="SUP", details_str="regmask_refimg")
# vmf.plot_oneimagemask(fixed_mask_nii, reg_mask_nii, patient_str = mov+"mixed", file_str ="SUP", details_str="regmask_refmask")
# # Registration for all
# rgf.register_all_frames()

# DICE 
# results_list = rgf.dice_all_patients()
# stats = rgf.stats_dice(results_list)
# print(results_list, stats)