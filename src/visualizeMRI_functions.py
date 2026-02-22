# src/visualizeMIR_functions.py
"""
Functions to visualize MRI images
"""

import numpy as np
from nilearn import image
from nilearn.plotting import plot_stat_map, show
import nibabel as nib # to get the nii format
from nibabel.affines import apply_affine

from paths import *

# Get single epoch 
def get_oneepoch(nii_obj,epoch_toplot):
    """
    """
    nii_obj_1t = image.index_img(nii_obj, epoch_toplot)
    return nii_obj_1t

# Plot 1 epoch 
def plot_oneepoch(nii_obj_1t, patient_str, print_infos=False, epoch_str = "", details_str=""):
    """
    """
    # Plot 3D image (x,y,z,) for this epoch
    plot_stat_map(
        nii_obj_1t,
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