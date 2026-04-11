# scripts/pca_eachpatient_script.py
"""
Script for the functions for the PCA "each patient"
"""

from src.models import pca_spatial as pcs
from src.models import pca as pc

# User preferences 
source_folder = "registered_framesBIS"
pca_description = "REGvoxROI"
maskYS = False 
maskbin = False
imageROIonly = True
pca_folder = "pca_allpatients_res"
original_shape = (128,128,32)
max_pc_calc = 150
pc_n1, pc_n2 = 0,1
eigenvectors_toplot = 10 

# LOAD VECTOR ARRAY AND PCA
X = pcs.get_vectorsarray(source_folder, pca_folder, details_str = pca_description, mask=maskYS, binary_mask=maskbin, image_roi_only=imageROIonly, recalculate= False)
pca, X_pca, meta = pcs.pca_patients(X, pca_folder, pca_description, normalize_rows=not maskbin, recalculate = True, max_pc_calc = max_pc_calc)

# EXPLAINED VARIANCE AND PLOT VALUES IN EIGENBASE
pc.plot_pca_explipower(pca, "allpatients_"+ pca_description)
pc.plot_pcvalues_2d(X_pca, pc_n1, pc_n2, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pc.plot_pcvalues_2d(X_pca, 2, 3, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pc.plot_pcvalues_2d(X_pca, 4, 5, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pc.plot_pcvalues_2d(X_pca, 6, 7, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pc.plot_pcvalues_2d(X_pca, 8, 9, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)

# CORRELATIONS WITH PATIENT METADATA
pcs.plot_pca_patientmeta(X_pca, pc_n1, pc_n2)
pcs.plot_pca_patientmeta(X_pca, 2, 3)
pcs.plot_pca_patientmeta(X_pca, 4, 5)
pcs.plot_pca_patientmeta(X_pca, 6, 7)
pcs.plot_pca_patientmeta(X_pca, 8, 9)

# PLOT EIGENVECTORS
pcs.plot_eigenvectors(X, pca, original_shape, pca_description, eigenvectors_toplot=min(eigenvectors_toplot, max_pc_calc))







