# src/pca_eachpatient_functions.py
"""
Script for the functions for the PCA "each patient"
"""

from paths import *
import pca_eachpatient_functions as pef 
import pca_functions as pf 

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
X = pef.get_vectorsarray(source_folder, pca_folder, details_str = pca_description, mask=maskYS, binary_mask=maskbin, image_roi_only=imageROIonly, recalculate= False)
pca, X_pca, meta = pef.pca_patients(X, pca_folder, pca_description, normalize_rows=not maskbin, recalculate = True, max_pc_calc = max_pc_calc)

# EXPLAINED VARIANCE AND PLOT VALUES IN EIGENBASE
pf.plot_pca_explipower(pca, "allpatients_"+ pca_description)
pf.plot_pcvalues_2d(X_pca, pc_n1, pc_n2, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pf.plot_pcvalues_2d(X_pca, 2, 3, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pf.plot_pcvalues_2d(X_pca, 4, 5, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pf.plot_pcvalues_2d(X_pca, 6, 7, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)
pf.plot_pcvalues_2d(X_pca, 8, 9, "allpatients_"+ pca_description, "_pc_in_eigenbase", scale_str ='Patient number', segments=False, axisscale_fixed=False)

# CORRELATIONS WITH PATIENT METADATA
pef.plot_pca_patientmeta(X_pca, pc_n1, pc_n2)
pef.plot_pca_patientmeta(X_pca, 2, 3)
pef.plot_pca_patientmeta(X_pca, 4, 5)
pef.plot_pca_patientmeta(X_pca, 6, 7)
pef.plot_pca_patientmeta(X_pca, 8, 9)

# PLOT EIGENVECTORS
pef.plot_eigenvectors(X, pca, original_shape, pca_description, eigenvectors_toplot=min(eigenvectors_toplot, max_pc_calc))







