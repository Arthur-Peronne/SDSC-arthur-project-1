# src/image_treatment.py
"""
Script to treat MRI images before analysis (PCA)
"""

import nibabel as nib
# from nilearn.image import resample_img
# from nilearn.image import resample_to_img
import numpy as np 
import glob
from pathlib import Path


from paths import *
import visualizeMRI_functions as vmf
import imagetreatment_functions as itf 

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "training/" #  "testing/" or "training/"
patient_name_1 = "patient003"
file_name_1 = "frame15"
epoch_toplot = 0
plot_onecropt = False 
cropandsave_all = False 

### BASIC RESAMPLING (tests)

# reference_path= path_datadir + "training/patient001/patient001_4d.nii.gz" # For reference
# target_path = path_datadir + datatype_tochoose_1 + patient_name_1 + "/" +  patient_name_1 + "_4d.nii.gz"  # For image
# reference_img = nib.load(reference_path)
# target_img = nib.load(target_path)
# test = itf.resample_basic(target_img, reference_img, patient_name_1, savefile=True, plotimage=True)
# print(test.shape)

# CROPPING FROM HEART MASK (ONE IMAGE)

if plot_onecropt:
    path_file = path_datadir + datatype_tochoose_1 +  patient_name_1 + "/" + patient_name_1 + "_" + file_name_1
    path_img = path_file + ".nii.gz"
    path_mask = path_file + "_gt.nii.gz"
    nii_img = nib.load(path_img)
    nii_mask = nib.load(path_mask)
    nii_cropped = itf.crop_heartzone_oneimage(nii_img, nii_mask)
    print(nii_cropped.shape)
    vmf.plot_oneepoch(nii_cropped, patient_name_1,oldstyle=False, epoch_str = "_" + file_name_1 + "_", details_str= "cropped")

# CROPPING ALL IMAGES AND PLOT THE CROPPED NII FILES 

if cropandsave_all:
    all_img, all_gt = itf.loaddata_tocrop()
    itf.crop_heartzone_allpatients()
    itf.plot_cropped_files()

# Checks on crop data 

# patient1, patient2 = "149", "124"
# path_file = path_tempodata_folder  + "cropped_nii/patient"+patient1+"_frame01.nii_cropped.nii.gz"
# path_file2 = path_tempodata_folder  + "cropped_nii/patient"+patient2+"_frame01.nii_cropped.nii.gz"
# img1 = nib.load(path_file)
# img2 = nib.load(path_file2)
patient1 = "099"
path_file = path_datadir  + "training/patient" + patient1 + "/patient" + patient1 + "_frame01_gt.nii.gz"
img1= nib.load(path_file)
data_3d = img1.get_fdata()
print(np.unique(data_3d))
# path_file2 = path_tempodata_folder  + "cropped_nii/patient" + patient1 + "_frame01.nii_cropped.nii.gz"
# img2= nib.load(path_file2)
# data_3d2= img2.get_fdata()
# print(np.unique(data_3d2))

print(img1.header)

def bbox_from_masked_nan(img, filtered="zeros"):
    """
    data_3d: numpy array (X,Y,Z) with NaN outside ROI, finite inside.
    Returns (xmin, xmax, ymin, ymax, zmin, zmax) and sizes (Nx,Ny,Nz).

    filtered="zeros" or "Nans"
    """
    data_3d = img.get_fdata()

    if filtered == "zeros":
        roi = (data_3d>0)
    else :
        roi = np.isfinite(data_3d)

    coords = np.where(roi)
    xmin, xmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    zmin, zmax = coords[2].min(), coords[2].max()

    Nx = xmax - xmin + 1
    Ny = ymax - ymin + 1
    Nz = zmax - zmin + 1

    return (xmin, xmax, ymin, ymax, zmin, zmax), (Nx, Ny, Nz)

# data_3d = img1.get_fdata()
# print(np.unique(data_3d))
bbox, size = bbox_from_masked_nan(img1)
print("bbox:", bbox)
print("size (Nx,Ny,Nz):", size)
# bbox, size = bbox_from_masked_nan(img2, filtered="nans")
# print("bbox:", bbox)
# print("size (Nx,Ny,Nz):", size)

gt = np.asanyarray(img1.dataobj)
cropped = gt[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1, bbox[4]:bbox[5]+1]
cropped_nii = nib.Nifti1Image(cropped, img1.affine, img1.header)

from nilearn.image import resample_img

# target_affine = cropped_nii.affine
# target_shape=(96,96,12)
img_res = resample_img(
    cropped_nii,
    target_affine=cropped_nii.affine,
    target_shape=(96,96,12),
    interpolation="nearest"
)
bbox, size = bbox_from_masked_nan(img_res)
print("bbox:", bbox)
print("size (Nx,Ny,Nz):", size)


# def centroid_from_cropped_nan(img_nii):
#     """
#     Heart voxels = finite voxels (not NaN).
#     Returns centroid in voxel indices and in world coordinates.
#     """
#     data = img_nii.get_fdata()
#     heart = np.isfinite(data)  # True where heart exists

#     ijk = np.column_stack(np.where(heart))          # (N, 3) voxel indices
#     centroid_voxel = ijk.mean(axis=0)               # (3,)

#     centroid_world = nib.affines.apply_affine(
#         img_nii.affine, centroid_voxel
#     )  # (3,)

#     return centroid_voxel, centroid_world, ijk.shape[0]

# cvox1, cworld1, n1 = centroid_from_cropped_nan(img1)
# cvox2, cworld2, n2 = centroid_from_cropped_nan(img2)

# print(f"Patient {patient1}: N(heart voxels)={n1}")
# print("  centroid voxel:", cvox1)
# print("  centroid world:", cworld1)

# print(f"Patient {patient2}: N(heart voxels)={n2}")
# print("  centroid voxel:", cvox2)
# print("  centroid world:", cworld2)

# print("\nDifferences:")
# print("  Δ voxel  :", cvox2 - cvox1, "  |Δ| =", np.linalg.norm(cvox2 - cvox1))
# print("  Δ world  :", cworld2 - cworld1, "  |Δ| =", np.linalg.norm(cworld2 - cworld1))
