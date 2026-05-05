# scripts/registration_script.py
"""
Script to perform the image resampling and the following registration
"""

# import nibabel as nib
# import numpy as np

# from src.config import TEMPODATA_FOLDER
from src.data import registration as rgt
from src.data import geometry as rgg
from src.data import resampling as rsp
# from src.visualization import mri_plots as mrp


only01 = False
resample_all = False
target_spacing = [1.5, 1.5, 3.15]
crop_all = False
register_all = True

# Main loops 

if resample_all:
    rsp.resample_all(target_spacing, only01 = only01)
if crop_all:
    rgg.crop_all_frames(only01 = only01)
if register_all:
    rgt.register_all_frames()