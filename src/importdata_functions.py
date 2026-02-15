# src/importdata_functions.py
"""
Functions to import data
"""

import nibabel as nib # to import the files

from paths import *

# Parameters : 
datatype_tochoose_1 = "testing/" # or "training/"
patient_name_1 = "patient102"
file_name_1 = "4d"

def extract_nii_file(datatype_tochoose, patient_name, file_name, print_infos=False):
    """
    Find the nii.gz file to extract, and return the nii object
    """
    # Find file with paths+name
    path_toextract = path_datadir + datatype_tochoose 
    name_toextract = patient_name + "/" +  patient_name + "_" + file_name + ".nii.gz"
    # Extract file 
    nii_obj= nib.load(path_toextract+name_toextract)
    # Optional: print infos 
    if print_infos:
        print(path_toextract+name_toextract)
        print(nii_obj.shape)
    return nii_obj

def convert_nii_file(nii_obj):
    """
    Info a Numpy array 
    """
    data_array = nii_obj.get_fdata()
    return data_array

# datatype_tochoose_1 = "testing/" # or "training/"
# patient_name_1 = "patient103"
# test = extract_nii_file(datatype_tochoose_1, patient_name_1, "4d", print_infos=False)
# b = convert_nii_file(test)
# # print(b)
# print(type(b))