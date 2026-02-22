# src/visualizeMIR_script.py
"""
Script to visualize MRI images
"""

import importdata_functions as idf
import visualizeMRI_functions as vmf

from paths import *

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "training/" #  "testing/" or "training/"
patient_name_1 = "patient002"
file_name_1 = "4d"
epoch_toplot = 0

# Extract nii object
# nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1, print_infos=True)

# Plot single or multiple epochs
# One epoch
# nii_obj_1t = vmf.get_oneepoch(nii_obj,epoch_toplot)
# vmf.plot_oneepoch(nii_obj_1t, patient_name_1, epoch_str = "_epoch0", details_str="")
# All epochs
# vmf.plot_allepochs(nii_obj, patient_name_1)

# Study of the position of the voxels 
# vmf.voxels_coordinates(nii_obj)
