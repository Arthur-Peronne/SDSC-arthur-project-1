# src/pca_functions.py
"""
Scripts to perform PCA
"""

from sklearn.decomposition import PCA

from paths import *
import importdata_functions as idf
import pca_functions as pf 

# USER ACTION: Choose patient and file (and epoch if single epoch to plot)
datatype_tochoose_1 = "testing/" # or "training/"
patient_name_1 = "patient103"

# Extract nii file and convert it to Numpy array
# filename_4d = path_datadir + datatype_tochoose_1 + patient_name_1 + "/" + patient_name_1 + "_4d.nii.gz"
nii_obj = idf.extract_nii_file(datatype_tochoose_1, patient_name_1, "4d", print_infos=False)
data_array = idf.convert_nii_file(nii_obj)

# PCA 1: each 3D image as a sample (how voxels co-vary over time, temporal dynamics) -> 30 lines, >100 000 columns (dimensions).

# Prepare data
X = pf.pca1_transpose(data_array, print_infos=False)
X_scaled = pf.pca_clean(X)
# PCA 
pca1 = PCA(n_components=nii_obj.shape[3])  # Get all principal components (all t, since it limits in this case)
X_reduced = pca1.fit_transform(X)
# Explained variance ratio: print + plot
print("Explained variance ratio:", pca1.explained_variance_ratio_)
pf.plot_pca_explipower(pca1, patient_name_1)






# PCA 2: each voxel as a sample (spatial modes, spatial patterns) -> >100 000 lines, 30 columns (dimensions).
