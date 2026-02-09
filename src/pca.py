# src/pca.py
"""
Script to perform PCA
"""

import nibabel as nib

# Extract data from all patients 

data_dir = "/home/renku/work/s3-bucket/ACDC/testing/"

all_volumes = []

for i in range(50):
    path_4d = data_dir + "/patient" + repr(i+101) + "/patient" + repr(i+101) + "_4d.nii.gz"
    data_extracted = nib.load(path_4d)
    all_volumes.append(data_extracted)

print(len(all_volumes))
print(all_volumes[0].shape)

