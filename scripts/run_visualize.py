# scripts/visualizeMRI_script.py
"""
Script to visualize MRI images
"""

import nibabel as nib

from src.config import TEMPODATA_FOLDER
from src.data import importdata as ipd
from src.visualization import mri_plots as mrp


# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "testing/"  # "testing/" or "training/"
patient_name_1 = "patient135"
file_name_1 = "frame01"  # "4d" or "frame01_gt" or "frame_1" or "frameXX_gt" or "frame_XX"
epoch_toplot = 0

# Extract nii object
# nii_obj = ipd.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1)
# nii_obj_mask = ipd.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1+"_gt")

# Plot single or multiple epochs
# One epoch
# if len(nii_obj.shape) > 3:
#     nii_obj_1t = mrp.get_oneepoch(nii_obj, epoch_toplot)
#     nii_obj_mask_1t = mrp.get_oneepoch(nii_obj_mask, epoch_toplot)
# else:
#     nii_obj_1t, nii_obj_mask_1t = nii_obj, nii_obj_mask
# mrp.plot_oneimg(nii_obj_1t, patient_str=patient_name_1, file_str=file_name_1, details_str="ORIGINAL")
# # Plot masks
# mrp.plot_onemask(nii_obj_mask_1t, patient_str=patient_name_1, file_str=file_name_1+"_gt", details_str="ORIGINAL")
# mrp.plot_oneimagemask(nii_obj_1t, nii_obj_mask_1t, patient_str=patient_name_1, file_str=file_name_1, details_str="ORIGINAL")

# All epochs
# mrp.plot_allepochs(nii_obj, patient_name_1)

# Study of the position of the voxels
# mrp.voxels_coordinates(nii_obj)

# Plot resampled images
# patient_name_2 = "patient035"
# file_name_2 = "frame01"
# path_img = TEMPODATA_FOLDER / "resampled_frames" / f"{patient_name_2}_{file_name_2}_resampled.nii.gz"
# path_gt = TEMPODATA_FOLDER / "resampled_frames" / f"{patient_name_2}_{file_name_2}_resampled_gt.nii.gz"
# nii_img = nib.load(path_img)
# nii_mask = nib.load(path_gt)
# mrp.plot_oneimg(nii_img, patient_str=patient_name_2, file_str=file_name_2, details_str="RESAMPLED")
# mrp.plot_onemask(nii_mask, patient_str=patient_name_2, file_str=file_name_2+"_gt", details_str="RESAMPLED")
# mrp.plot_oneimagemask(nii_img, nii_mask, patient_str=patient_name_2, file_str=file_name_2, details_str="RESAMPLED")

# Plot with mask
# path_img = TEMPODATA_FOLDER / "resampled_frames" / f"{patient_name_1}_{file_name_1}_resampled.nii.gz"
# path_mask = TEMPODATA_FOLDER / "resampled_frames" / f"{patient_name_1}_{file_name_1}_resampled_gt.nii.gz"
# mrp.plot_masks(path_img, path_mask, patient_name_1, file_name_1, plot_ref=True, plot_refandmask=True, plot_onlymasked=True)