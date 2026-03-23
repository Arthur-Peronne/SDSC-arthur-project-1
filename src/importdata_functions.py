# src/importdata_functions.py
"""
Functions to import data
"""

import nibabel as nib
import glob
import os 

from paths import *


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
    # training_files = glob.glob(path_datadir + "training/patient*/Info.cfg")
    # testing_files = glob.glob(path_datadir + "testing/patient*/Info.cfg")
    training_files = sorted(glob.glob(path_datadir + "training/patient*/Info.cfg"))
    testing_files = sorted(glob.glob(path_datadir + "testing/patient*/Info.cfg"))
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

import os

def get_patient_acdc_path(patient_id,  file_type="frame",  base_path="/home/renku/work/s3-bucket/ACDC", check_exists=False):
    """
    Return the path to one ACDC file for a given patient.

    Parameters
    ----------
    patient_id : int
        Patient number, from 1 to 150.
    file_type : str
        Type of file to retrieve:
        - "frame"    : initial frame NIfTI
        - "mask"     : GT mask of the initial frame
        - "4d"       : 4D NIfTI with all epochs
        Default: "frame"
    base_path : str
        Root folder of the ACDC dataset.
    check_exists : bool
        If True, raise FileNotFoundError when the path does not exist.

    Returns
    -------
    str
        Full path to the requested file.
    """

    if not (1 <= patient_id <= 150):
        raise ValueError("patient_id must be between 1 and 150")

    if file_type not in {"frame", "mask", "4d"}:
        raise ValueError("file_type must be one of: 'frame', 'mask', '4d'")

    subset = "training" if patient_id <= 100 else "testing"
    patient_str = f"patient{patient_id:03d}"

    # Special case: patient 90 uses frame04 instead of frame01
    frame_num = 4 if patient_id == 90 else 1
    frame_str = f"frame{frame_num:02d}"

    if file_type == "frame":
        filename = f"{patient_str}_{frame_str}.nii.gz"
    elif file_type == "mask":
        filename = f"{patient_str}_{frame_str}_gt.nii.gz"
    else:  # file_type == "4d"
        filename = f"{patient_str}_4d.nii.gz"

    full_path = os.path.join(base_path, subset, patient_str, filename)

    if check_exists and not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    return full_path

def get_patient_modified_path(patient_id, folder, file_type="frame", base_path="/home/renku/work/SDSC-arthur-project-1/tempodata",check_exists=False):
    """
    Return the path to a modified (e.g. registered) NIfTI file for a given patient.

    Parameters
    ----------
    patient_id : int
        Patient number (1 to 150)
    folder : str
        Subfolder inside tempodata (e.g. "registered_framesBIS")
    file_type : str
        Type of file:
        - "frame" : modified image (default)
        - "mask"  : modified GT mask
    base_path : str
        Root path to tempodata
    check_exists : bool
        If True, raise error if file does not exist

    Returns
    -------
    str
        Full path to the requested file
    """

    if not (1 <= patient_id <= 150):
        raise ValueError("patient_id must be between 1 and 150")

    if file_type not in {"frame", "mask"}:
        raise ValueError("file_type must be 'frame' or 'mask'")

    patient_str = f"patient{patient_id:03d}"

    # Special case: patient 90 uses frame04
    frame_num = 4 if patient_id == 90 else 1
    frame_str = f"frame{frame_num:02d}"

    # Build filename
    if file_type == "frame":
        filename = f"{patient_str}_{frame_str}_registered.nii.gz"
    else:  # mask
        filename = f"{patient_str}_{frame_str}_registered_gt.nii.gz"

    full_path = os.path.join(base_path, folder, filename)

    if check_exists and not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    return full_path