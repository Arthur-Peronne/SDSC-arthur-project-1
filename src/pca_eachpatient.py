# src/pca_eachpatient.py
"""
Script to perform PCA with images from each patient (spatial PCA, not temporal anymore)
"""

import glob
from joblib import Parallel, delayed
import nibabel as nib # to import the files
import numpy as np

from paths import *
import importdata_functions as idf
import image_treatment as igt 

# USER ACTION:

# Data to work from 
timeframe = 0
reference_path = path_datadir + "training/patient001/patient001_4d.nii.gz" # For reference
reload_files = False

if reload_files:
    # Get all paths to nii files
    training_files = glob.glob(path_datadir + "training/patient*/patient*_4d.nii.gz")
    testing_files = glob.glob(path_datadir + "testing/patient*/patient*_4d.nii.gz")
    all_files = training_files + testing_files # and optional: all_files.sort() 
    # Create ref obj for resampling 
    nii_ref = nib.load(reference_path)
    # Epoch 0 extraction from the nii file to be searched from all_files, and resampling (256x256x10)
    def load_and_extract_epoch_zero(file_path, timeframe=0):
        # Load the NIfTI file
        nii_obj = nib.load(file_path)
        # Extract t0 data and create a tempo nii_file with it
        epoch_data = np.asanyarray(nii_obj.dataobj[..., timeframe], dtype=np.float32)
        nii_obj_t = nib.Nifti1Image(epoch_data, nii_obj.affine, nii_obj.header)
        # Resample to put it in correct resolution
        # Return the numpy array 
        nii_obj_res = igt.resample_basic(nii_obj_t, nii_ref)
        return np.asanyarray(nii_obj_res.dataobj, dtype=np.float32)
    # Use parallel processing to load and extract epoch zero
    img_list = Parallel(n_jobs=-1)(delayed(load_and_extract_epoch_zero)(file, timeframe) for file in all_files)
    # for i, img in enumerate(img_list):
    #     print(f"Image {i}: {img.shape}")
    # Convert the list of 3D images to a single 4D NumPy array
    img_array = np.stack(img_list)
    # print(img_array.shape)  # Should be (150, depth, height, width)
    out_path = path_tempodata_folder  + "nparraydata_for_pcaeachpatient.npy"
    np.save(out_path, img_array)

else:
    out_path = path_tempodata_folder  + "nparraydata_for_pcaeachpatient.npy"
    img_array = np.load(out_path)
    print(img_array.shape, img_array.dtype)

    