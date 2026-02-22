# src/image_treatment.py
"""
Script to treat MRI images before analysis (PCA)
"""

import nibabel as nib
from nilearn.image import resample_img
from nilearn.image import resample_to_img
import numpy as np 

from paths import *
import visualizeMRI_functions as vmf

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "training/" #  "testing/" or "training/"
patient_name_1 = "patient002"
# file_name_1 = "4d"
epoch_toplot = 0


### BASIC RESAMPLING 

# Function

def resample_basic(target_img, reference_img, target_shape = (256, 256, 10), savefile=False, plotimage=False):
    # # Load files 
    # reference_img = nib.load(reference_path)
    # target_img = nib.load(target_path)
    # Resample (just interpolate  to correct shape)
    resampled_img = resample_img(
        target_img,
        target_affine=reference_img.affine,
        target_shape=target_shape,
        interpolation='nearest')
    # Save modified imaged
    if savefile:
        resampled_path = path_resultsfolder +  patient_name_1 + "_4d_RESAMPLED001.nii.gz" 
        nib.save(resampled_img, resampled_path)
    # Plot 
    if plotimage:
        target_img_1t = vmf.get_oneepoch(target_img, epoch_toplot)
        vmf.plot_oneepoch(target_img_1t, patient_name_1, epoch_str = "_epoch0", details_str="")
        resampled_img_1t = vmf.get_oneepoch(resampled_img, epoch_toplot)
        vmf.plot_oneepoch(resampled_img_1t, patient_name_1, epoch_str = "_epoch0", details_str="RESAMPLED001")
    return resampled_img


# def resample_basic_3d(target_img_3d, reference_img_3d, target_shape = (256, 256, 10), interpolation="linear"):
#     """
#     Same but with 3D images, to help for pca_eachpatient
#     """
#     return resample_to_img(
#         target_img_3d,
#         reference_img_3d,
#         interpolation=interpolation
#     )


# Test
reference_path= path_datadir + "training/patient001/patient001_4d.nii.gz" # For reference
target_path = path_datadir + datatype_tochoose_1 + patient_name_1 + "/" +  patient_name_1 + "_4d.nii.gz"  # For image
reference_img = nib.load(reference_path)
target_img = nib.load(target_path)
# test = resample_basic(target_img, reference_img, savefile=True, plotimage=True)
# print(type(test))
# print(test.shape)

# reference_img_t0 = nib.Nifti1Image(np.asanyarray(reference_img.dataobj[..., 0], dtype=np.float32), reference_img.affine, reference_img.header)
# target_img_t0 = nib.Nifti1Image(np.asanyarray(target_img.dataobj[..., 0], dtype=np.float32), target_img.affine, target_img.header)
# testbis = resample_basic_3d(target_img, reference_img)
# print(type(testbis))
# print(testbis.shape)

# # Load reference image -> Patient 001
# reference_path= path_datadir + "training/patient001/patient001_4d.nii.gz" # For reference
# reference_img = nib.load(reference_path)
# # Load other image to align 
# target_path = path_datadir + datatype_tochoose_1 + patient_name_1 + "/" +  patient_name_1 + "_4d.nii.gz"  # For image
# target_img = nib.load(target_path)
# # Resample (just interpolate  to correct shape)
# target_shape = (256, 256, 10) # Define a common target shape (e.g., (256, 256, 10))
# resampled_img = resample_img(
#     target_img,
#     target_affine=reference_img.affine,
#     target_shape=target_shape,
#     interpolation='nearest')
# # Save modified imaged
# resampled_path = path_resultsfolder +  patient_name_1 + "_4d_RESAMPLED001.nii.gz" 
# nib.save(resampled_img, resampled_path)
# # Checks 
# # print(target_img.shape)
# # print(resampled_img.shape)
# # Plot 
# target_img_1t = vmf.get_oneepoch(target_img, epoch_toplot)
# vmf.plot_oneepoch(target_img_1t, patient_name_1, epoch_str = "_epoch0", details_str="")
# resampled_img_1t = vmf.get_oneepoch(resampled_img, epoch_toplot)
# vmf.plot_oneepoch(resampled_img_1t, patient_name_1, epoch_str = "_epoch0", details_str="RESAMPLED001")


### COMPLEX IMAGE TRANSFORMATION -> IN PROGRESS

# import SimpleITK as sitk
# import numpy as np
# import tempfile
# import os

# # Image and  alignment 
# # Load reference image -> Patient 001
# reference_path= path_datadir + "training/patient001/patient001_4d.nii.gz" # For reference
# # reference_img = nib.load(reference_path)
# # reference_img = nib.as_closest_canonical(reference_img) # Canonisation -> dims right direction
# # reference_img_t0 = nib.Nifti1Image(reference_img.get_fdata()[..., 0],reference_img.affine)
# # # Load other image to align 
# target_path = path_datadir + datatype_tochoose_1 + patient_name_1 + "/" +  patient_name_1 + "_4d.nii.gz"  # For image
# # target_img = nib.load(target_path)
# # target_img = nib.as_closest_canonical(target_img) # Canonisation -> dims right direction
# # target_img_to = nib.Nifti1Image(target_img.get_fdata()[..., 0],target_img.affine)
# # Canonisation 
# template_nib = nib.load(reference_path)
# template_nib = nib.as_closest_canonical(template_nib)
# target_nib = nib.load(target_path)
# target_nib = nib.as_closest_canonical(target_nib)


# with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_fixed, \
#      tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_moving:

#     fixed_temp_path = tmp_fixed.name
#     moving_temp_path = tmp_moving.name

# # Sauvegarde temporaire
# nib.save(template_nib, fixed_temp_path)
# nib.save(target_nib, moving_temp_path)

# try:
#     # --- Lecture ITK ---
#     fixed = sitk.ReadImage(fixed_temp_path)
#     moving = sitk.ReadImage(moving_temp_path)

#     # Extraire t=0
#     fixed = fixed[:, :, :, 0]
#     moving = moving[:, :, :, 0]

#     fixed = sitk.Cast(fixed, sitk.sitkFloat32)
#     moving = sitk.Cast(moving, sitk.sitkFloat32)

#     # --- Registration ---
#     registration = sitk.ImageRegistrationMethod()
#     registration.SetMetricAsMattesMutualInformation(50)
#     registration.SetOptimizerAsGradientDescent(
#         learningRate=1.0,
#         numberOfIterations=200,
#         convergenceMinimumValue=1e-6,
#         convergenceWindowSize=10
#     )
#     registration.SetInterpolator(sitk.sitkLinear)

#     initial_transform = sitk.CenteredTransformInitializer(
#         fixed, moving,
#         sitk.Euler3DTransform(),
#         sitk.CenteredTransformInitializerFilter.GEOMETRY
#     )

#     registration.SetInitialTransform(initial_transform)

#     final_transform = registration.Execute(fixed, moving)

#     aligned = sitk.Resample(
#         moving,
#         fixed,
#         final_transform,
#         sitk.sitkLinear,
#         0.0,
#         moving.GetPixelID()
#     )

# finally:
#     # --- Suppression des fichiers temporaires ---
#     os.remove(fixed_temp_path)
#     os.remove(moving_temp_path)


# # # ITK 
# # fixed = sitk.ReadImage(reference_path)
# # moving = sitk.ReadImage(target_path)
# # fixed_sitk = fixed[:, :, :, 0]
# # moving_sitk = moving[:, :, :, 0]
# # fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
# # moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
# # # Registration 
# # registration = sitk.ImageRegistrationMethod()

# # registration.SetMetricAsMattesMutualInformation(50)
# # registration.SetOptimizerAsGradientDescent(
# #     learningRate=1.0,
# #     numberOfIterations=200,
# #     convergenceMinimumValue=1e-6,
# #     convergenceWindowSize=10
# # )

# # registration.SetInterpolator(sitk.sitkLinear)

# # initial_transform = sitk.CenteredTransformInitializer(
# #     fixed_sitk,
# #     moving_sitk,
# #     sitk.Euler3DTransform(),
# #     sitk.CenteredTransformInitializerFilter.GEOMETRY
# # )

# # registration.SetInitialTransform(initial_transform)

# # final_transform = registration.Execute(fixed_sitk, moving_sitk)
# # print("Optimizer stop condition:", registration.GetOptimizerStopConditionDescription())
# # print("Final metric value:", registration.GetMetricValue())
# # #  Rasample
# # aligned = sitk.Resample(
# #     moving_sitk,
# #     fixed_sitk,
# #     final_transform,
# #     sitk.sitkLinear,
# #     0.0,
# #     moving_sitk.GetPixelID()
# # )
# # # Transformation in Nitfi again 
# aligned_array = sitk.GetArrayFromImage(aligned)  # (z, y, x)

# # Remettre en (x, y, z)
# aligned_array = np.transpose(aligned_array, (2, 1, 0))

# # On récupère l'affine du template canonicalisé
# template_affine = template_nib.affine

# aligned_nifti = nib.Nifti1Image(
#     aligned_array,
#     template_affine
# )


# # print(aligned_nifti)
# # print(type(aligned_nifti))

# vmf.plot_oneepoch(aligned_nifti, patient_name_1, epoch_str = "_epoch0", details_str="ALIGNED001")





### OLD 





# print(nii_obj.shape)
# # nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1, print_infos=True)
# nii_obj_1t = vmf.get_oneepoch(nii_obj,epoch_toplot)
# vmf.plot_oneepoch(nii_obj_1t, patient_name_1, epoch_str = "_epoch0", details_str="ALI-RESA001")
# # Original 
# nii_objori= nib.load(target_path)
# print(nii_objori.shape)
# nii_obj_1tori = vmf.get_oneepoch(nii_objori,epoch_toplot)
# vmf.plot_oneepoch(nii_obj_1tori, patient_name_1, epoch_str = "_epoch0", details_str="")



# print("Fixed direction:", fixed_sitk.GetDirection())
# print("Fixed spacing:", fixed_sitk.GetSpacing())
# print("Fixed origin:", fixed_sitk.GetOrigin())

# print(template_nib.affine)

# sitk.WriteImage(aligned, "patient102_aligned.nii.gz")


# target_path = path_datadir + datatype_tochoose_1 + patient_name_1 + "/" +  patient_name_1 + "_4d.nii.gz"  # For image
# target_img = nib.load(target_path)
# # Axis canonisation

# print("Avant :", nib.aff2axcodes(reference_img.affine))
# img_canonical = nib.as_closest_canonical(reference_img)
# print("Après :", nib.aff2axcodes(img_canonical.affine))
# print("Nouvelle shape :", img_canonical.shape)


# # # Align
# # aligned_img, _ = align(
# #     target_img,
# #     reference_img,
# #     resample_target=reference_img  # This ensures the output matches the reference's space
# # )
# # # Resample
# # target_shape = (256, 256, 10) # Define a common target shape (e.g., (256, 256, 10))
# # registered_img = resample_img(
# #     aligned_img,
# #     target_affine=reference_img.affine,
# #     target_shape=target_shape,
# #     interpolation='nearest'
# )
# # Save modified imaged
# aligned_path = path_resultsfolder +  patient_name_1 + "_4d_ALI-RESA001.nii.gz" 
# nib.save(registered_img, aligned_path)
# # registered_images.append(registered_img)
# # reference_img = nib.load("path/to/reference_image.nii.gz")

# # reference_path= path_datadir + "training/patient001/patient001_4d.nii.gz" 
# nii_obj = nib.load(aligned_path)
# print(nii_obj.shape)
# # nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1, print_infos=True)
# nii_obj_1t = vmf.get_oneepoch(nii_obj,epoch_toplot)
# vmf.plot_oneepoch(nii_obj_1t, patient_name_1, epoch_str = "_epoch0", details_str="ALI-RESA001")
# # Original 
# nii_objori= nib.load(target_path)
# print(nii_objori.shape)
# nii_obj_1tori = vmf.get_oneepoch(nii_objori,epoch_toplot)
# vmf.plot_oneepoch(nii_obj_1tori, patient_name_1, epoch_str = "_epoch0", details_str="")