# src/visualizeMIR_script.py
"""
Script to visualize MRI images
"""

import importdata_functions as idf
import visualizeMRI_functions as vmf

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "testing/" # or "training/"
patient_name_1 = "patient103"
file_name_1 = "4d"
epoch_toplot = 10

# Extract nii object
nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, file_name_1, print_infos=True)

# Plot single or multiple epochs
# One epoch
# nii_obj_1t = vmf.get_oneepoch(nii_obj,epoch_toplot)
# vmf.plot_oneepoch(nii_obj_1t, patient_name_1, epoch_number =  epoch_toplot, print_infos=True)
# All epochs
vmf.plot_allepochs(nii_obj, patient_name_1)
