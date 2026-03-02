# src/visualizeMIR_functions.py
"""
Functions to visualize MRI images
"""

import numpy as np
from nilearn import image
from nilearn.image import resample_to_img
from nilearn import plotting
from nilearn.plotting import plot_stat_map, show
import nibabel as nib 
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt

from paths import *
import importdata_functions as idf

# Get single epoch 
def get_oneepoch(nii_obj,epoch_toplot):
    """
    """
    nii_obj_1t = image.index_img(nii_obj, epoch_toplot)
    return nii_obj_1t

# Plot 1 epoch 
def plot_oneepoch(nii_obj_1t, patient_str, oldstyle=True, print_infos=False, epoch_str = "", details_str="", cl_yesno = False, cl = (0,400)):
    """
    """
    # Plot 3D image (x,y,z,) for this epoch
    if oldstyle == True:
        if cl_yesno:
                plot_stat_map(
                nii_obj_1t,
                vmin=cl[0],
                vmax=cl[1],
                title= patient_str + " " + epoch_str + " " + details_str,
                output_file=path_resultsfolder + patient_str + epoch_str + details_str + ".png"
            )
        else:
            plot_stat_map(
                nii_obj_1t,
                title= patient_str + " " + epoch_str + " " + details_str,
                output_file=path_resultsfolder + patient_str + epoch_str + details_str + ".png"
            )
    else : 
        if cl_yesno:
                plot_stat_map(
                nii_obj_1t,
                cmap = "plasma",
                vmin=cl[0],
                vmax=cl[1],
                title= patient_str + " " + epoch_str + " " + details_str,
                output_file=path_resultsfolder + patient_str + epoch_str + details_str + ".png"
            )
        else:
            plot_stat_map(
                nii_obj_1t,
                cmap = "plasma",
                title= patient_str + " " + epoch_str + " " + details_str,
                output_file=path_resultsfolder + patient_str + epoch_str + details_str + ".png"
            )
    if print_infos:
        print("Saved epoch "+ repr(epoch_number))

# Plot all epochs t
def plot_allepochs(nii_obj, patient_name,  print_infos=True, epoch_limit=1000, suffix=""):
    """
    """
    # Get number of epochs
    n_epochs = nii_obj.shape[3]
    # Plot 3D image (x,y,z,) for every epoch
    for t in range(min(n_epochs, epoch_limit)):
        vol = image.index_img(nii_obj, t)
        plot_stat_map(
            vol,
            title=patient_name + f" – epoch {t}",
            output_file=path_resultsfolder + patient_name + "_epoch_" +repr(t)+ suffix + ".png"
        )
        if print_infos:
            print(f"Saved epoch {t}")

def from_longvec_to_image(vec, original_shape, nii_obj, patient_name, suffix_1):
    """
    Get long np vector, reshape it , transform it into nii and plot it 
    """
    vec_3d = vec.reshape(original_shape)
    nii_file = nib.Nifti1Image(vec_3d, nii_obj.affine, nii_obj.header)
    plot_oneepoch(nii_file, patient_name, details_str=suffix_1)

def voxels_coordinates(nii_obj):
    """
    """
    affine = nii_obj.affine
    shape = nii_obj.shape[:3]
    i, j, k = np.indices(shape) # Create a grid of voxel indices (i, j, k)
    voxel_indices = np.vstack([i.ravel(), j.ravel(), k.ravel()]).T # Stack the indices into a 3xN array
    voxel_coordinates = apply_affine(affine, voxel_indices) # Apply the affine transformation to get the real-world coordinates
    # Extract x, y, and z coordinates
    x_coords = voxel_coordinates[:, 0]
    y_coords = voxel_coordinates[:, 1]
    z_coords = voxel_coordinates[:, 2]
    # Get unique positions along each axis
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    unique_z = np.unique(z_coords)
    print("Unique x positions:", unique_x)
    print("Unique y positions:", unique_y)
    print("Unique z positions:", unique_z)

def plot_masks(datatype_tochoose, patient_name, file_name_img, plot_ref=True, plot_refandmask=True, plot_onlymasked=True):
    """
    """
    # Extract nii object and reference
    nii_obj_img = idf.extract_nii_file(datatype_tochoose, patient_name, file_name_img)
    file_name_mask = file_name_img + "_gt" # "4d" or "frame01_gt" or "frame_1" or "frameXX_gt" or "frame_XX"
    nii_obj_mask = idf.extract_nii_file(datatype_tochoose, patient_name, file_name_mask, print_infos=True)

    # Plot img for reference 
    # Scale
    if plot_ref or plot_refandmask:
        img_data = nii_obj_img.get_fdata()
        finite = np.isfinite(img_data)
        vmin, vmax = np.percentile(img_data[finite], (2, 98))  
        # Plot and save 
        display = plotting.plot_anat(
            nii_obj_img,
            title= file_name_img + " and mask",
            vmin=vmin, vmax=vmax,
        )
        if plot_ref:
            display.savefig(path_resultsfolder + patient_name + "_" + file_name_img + "_raw" + ".png")
    # Plot img + mask 
        if plot_refandmask:
            mask_r = resample_to_img(nii_obj_mask, nii_obj_img, interpolation="nearest")
            display.add_overlay(mask_r, transparency=0.5)
            display.savefig(path_resultsfolder + patient_name + "_" + file_name_img + "_superposition" + ".png")
        display.close()
    # Plot only elements in mask 
    if plot_onlymasked:
        mask_data = nii_obj_mask.get_fdata()   # returns a float64 numpy array (X,Y,Z) or (X,Y,Z,T)
        region = (mask_data != 0)
        filtered_nan = img_data.copy()
        filtered_nan[~region] = np.nan
        filtered_img = nib.Nifti1Image(filtered_nan, affine=nii_obj_img.affine)
        display_masked = plotting.plot_anat(
            filtered_img,
            title= file_name_img + " region of interest (heart) only",
            vmin=vmin, vmax=vmax)
        display_masked.savefig(path_resultsfolder + patient_name + "_" + file_name_img + "_onlymasked" +".png")
        display_masked.close()


def plot_one_t_bw(nii_img_1t, print_infos=False, patient_str ="", file_str = "", details_str=""):
    """
    """
    # Color limits 
    data_img = nii_img_1t.get_fdata()
    finite = np.isfinite(data_img)
    vmin, vmax = np.percentile(data_img[finite], (2, 98))  
    # Plot 3D 
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="white")   # NaNs -> white
    display = plotting.plot_anat(
        nii_img_1t,
        cmap="plasma",
        black_bg=True,
        vmin=vmin, vmax=vmax,
        title= patient_str + " " + file_str + " " + details_str)
    # Save 
    output_file = path_resultsfolder + patient_str + file_str + details_str + ".png"
    display.savefig(output_file)
    if print_infos:
        print("Image saved in " + output_file)