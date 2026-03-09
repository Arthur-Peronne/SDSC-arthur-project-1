# src/importdata_functions.py
"""
Functions to import data
"""

import nibabel as nib # to import the files
import glob

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

def read_info_cfg(filepath):
    """
    """
    info = {}
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            if not line or ":" not in line:
                continue  # skip empty lines
            
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to int or float if possible
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string
            
            info[key] = value
    
    return info

def load_allframes(only01 = False):
    """
    """   
    if only01: # Only frame01
        training_img = glob.glob(path_datadir + "training/patient*/patient*_frame01.nii.gz")
        training_img90 = glob.glob(path_datadir + "training/patient*/patient*_frame04.nii.gz") # For patient090
        testing_img  = glob.glob(path_datadir + "testing/patient*/patient*_frame01.nii.gz")
        all_img = training_img + training_img90 + testing_img
        training_gt = glob.glob(path_datadir + "training/patient*/patient*_frame01_gt.nii.gz")
        training_gt90 = glob.glob(path_datadir + "training/patient*/patient*_frame04_gt.nii.gz") # For patient090
        testing_gt  = glob.glob(path_datadir + "testing/patient*/patient*_frame01_gt.nii.gz")
        all_gt = training_gt + training_gt90 + testing_gt
    else: # Frame01 and FrameXX
        training_img = glob.glob(path_datadir + "training/patient*/patient*_frame[0-9][0-9].nii.gz")
        testing_img  = glob.glob(path_datadir + "testing/patient*/patient*_frame[0-9][0-9].nii.gz")
        all_img = training_img + testing_img
        training_gt = glob.glob(path_datadir + "training/patient*/patient*_frame*_gt.nii.gz")
        testing_gt  = glob.glob(path_datadir + "testing/patient*/patient*_frame*_gt.nii.gz")
        all_gt = training_gt + testing_gt

    return all_img, all_gt

def load_allframes_resampled(only01 = True):
    """
    """   
    if only01: # Only frame01
        all_img = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame01_resampled.nii.gz")
        all_gt = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame01_resampled_gt.nii.gz")
        img_90 = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame04_resampled.nii.gz")
        gt_90 = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame04_resampled_gt.nii.gz")
        all_img, all_gt = all_img+img_90, all_gt+gt_90
    else: # Frame01 and FrameXX
        all_img = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame[0-9][0-9]_resampled.nii.gz")
        all_gt = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame[0-9][0-9]_resampled_gt.nii.gz")

    return all_img, all_gt

def load_allgt_res(onlytraining=False):
    """
    """   
    if onlytraining:
        all_gt = glob.glob(path_tempodata_folder + "resampled_frames/patient0*_frame[0-9][0-9]_resampled_gt.nii.gz")
        all_gt += glob.glob(path_tempodata_folder + "resampled_frames/patient100_frame[0-9][0-9]_resampled_gt.nii.gz")
    else:
        all_gt = glob.glob(path_tempodata_folder + "resampled_frames/patient*_frame[0-9][0-9]_resampled_gt.nii.gz")
    return all_gt

def import_patientmetapaths(printinfos=True):
    # Load
    training_files = glob.glob(path_datadir + "training/patient*/Info.cfg")
    testing_files = glob.glob(path_datadir + "testing/patient*/Info.cfg")
    all_files = training_files + testing_files # and optional: all_files.sort() 
    if printinfos:
        print("First file:", all_files[0])
        print("Last file :", all_files[-1])
        print("Total     :", len(all_files))
    return all_files

def load_allcroppedframes():
    """
    """   
    all_img_crop = glob.glob(path_tempodata_folder + "cropped_frames/patient*_frame*_cropped.nii.gz")
    return sorted(all_img_crop)