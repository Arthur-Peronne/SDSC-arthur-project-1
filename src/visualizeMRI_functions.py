# src/visualizeMIR_functions.py
"""
Functions to visualize MRI images
"""

from nilearn import image
from nilearn.plotting import plot_stat_map, show

from paths import *

# Get single epoch 
def get_oneepoch(nii_obj,epoch_toplot):
    """
    """
    nii_obj_1t = image.index_img(nii_obj, epoch_toplot)
    return nii_obj_1t

# Plot 1 epoch 
def plot_oneepoch(nii_obj_1t, patient_name, epoch_number = 0, print_infos=False):
    """
    """
    # Plot 3D image (x,y,z,) for this epoch
    plot_stat_map(
        nii_obj_1t,
        title=patient_name + f" – epoch {epoch_number}",
        output_file=path_resultimagesfolder + patient_name + "_epoch_" + repr(epoch_number) +".png"
    )
    if print_infos:
        print("Saved epoch "+ repr(epoch_number))

# Plot all epochs t
def plot_allepochs(nii_obj, patient_name,  print_infos=True):
    """
    """
    # Get number of epochs
    n_epochs = nii_obj.shape[3]
    # Plot 3D image (x,y,z,) for every epoch
    for t in range(n_epochs):
        vol = image.index_img(nii_obj, t)
        plot_stat_map(
            vol,
            title=patient_name + f" – epoch {t}",
            output_file=path_resultimagesfolder + patient_name + "_epoch_"+repr(t)+".png"
        )
        if print_infos:
            print(f"Saved epoch {t}")

