# src/visualizeMIR_script.py
"""
Script to visualize MRI images
"""

import nibabel as nib 

import importdata_functions as idf
import visualizeMRI_functions as vmf
from paths import *

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "testing/" #  "testing/" or "training/"
patient_name_1 = "patient135"
file_name_1 = "frame01" # "4d" or "frame01_gt" or "frame_1" or "frameXX_gt" or "frame_XX"
epoch_toplot = 0

# Extract nii object
# nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1)
# nii_obj_mask = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1+"_gt")

#  Plot single or multiple epochs
# One epoch
# if len(nii_obj.shape) > 3:
#     nii_obj_1t = vmf.get_oneepoch(nii_obj,epoch_toplot)
#     nii_obj_mask_1t = vmf.get_oneepoch(nii_obj_mask,epoch_toplot)
# else:
#     nii_obj_1t, nii_obj_mask_1t = nii_obj, nii_obj_mask
# vmf.plot_oneimg(nii_obj_1t, patient_str =patient_name_1, file_str =file_name_1, details_str="ORIGINAL")
# # Plot masks
# vmf.plot_onemask(nii_obj_mask_1t, patient_str =patient_name_1, file_str =file_name_1+"_gt", details_str="ORIGINAL")
# vmf.plot_oneimagemask(nii_obj_1t, nii_obj_mask_1t, patient_str =patient_name_1, file_str =file_name_1, details_str="ORIGINAL")

# All epochs
# vmf.plot_allepochs(nii_obj, patient_name_1)

# Study of the position of the voxels 
# vmf.voxels_coordinates(nii_obj)

# # Plot resampled images 
# # Load 
# patient_name_2 = "patient035"
# file_name_2 = "frame01"
# path_img = path_tempodata_folder + "resampled_frames/" + patient_name_2 + "_" + file_name_2 + "_resampled.nii.gz"
# path_gt = path_tempodata_folder + "resampled_frames/" + patient_name_2 + "_" + file_name_2 + "_resampled_gt.nii.gz"
# nii_img = nib.load(path_img)
# nii_mask = nib.load(path_gt)
# # Plot 
# vmf.plot_oneimg(nii_img, patient_str =patient_name_2, file_str =file_name_2, details_str="RESAMPLED")
# vmf.plot_onemask(nii_mask, patient_str =patient_name_2, file_str =file_name_2+"_gt", details_str="RESAMPLED")
# vmf.plot_oneimagemask(nii_img, nii_mask, patient_str =patient_name_2, file_str =file_name_2, details_str="RESAMPLED")


# Plot with mask 
# patient_name = "010"
# file_name = "frame01"
# path_img = path_tempodata_folder + "resampled_frames/" + patient_name_1 + "_" + file_name_1 + "_resampled" + ".nii.gz"
# path_mask = path_tempodata_folder + "resampled_frames/" + patient_name_1 + "_" + file_name_1 + "_resampled_gt" + ".nii.gz"
# # vmf.plot_masks(datatype_tochoose_1, patient_name_1, file_name_1)
# vmf.plot_masks(path_img, path_mask, patient_name_1, file_name_1, plot_ref=True, plot_refandmask=True, plot_onlymasked=True)
