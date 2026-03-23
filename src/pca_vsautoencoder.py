# src/pca_eachpatient_functions.py
"""
Script for the functions for the PCA  across each patient and compare the results with the autoencoder
Reasons to have a new script: need same inputs and same outputs as AE pipeline
"""

import numpy as np 
import nibabel as nib 

from paths import *
import pca_eachpatient_functions as pef 
import autoencoder_functions as aef 

# User choices : DATA
n_training = 120
splitname = "split0"
# User choices : MODEL
latent_dimensions = 119
latdim_list = [1,2,3,4,5,7,10,15,20,30,40,50,80,119]
# User choices : RECONSTRUCTION 
patient_toplot = 144
check_ontrainset = False 

# Get data
X = pef.get_vectorsarray("registered_framesBIS", "X_vectors", details_str = "REGvoxROI", image_roi_only=True, recalculate= False)
X_train, X_test = X[:n_training], X[n_training:]
del X 

# Model training/loading 
pca_filename = "PCA_" + repr(n_training) + "patients_" + splitname
pca, X_train_pca, meta = pef.pca_patients(X_train, "autoencoder", pca_filename,  normalize_rows=False, recalculatePCA = False)
X_test_pca = pca.transform(X_test)

# # Reconstruct test patient from PCA and calculate results averaged over test sample
# for latent_dimensions in latdim_list:
#     all_metrics = []
#     for (i, x_patient_flat) in enumerate(X_test):
#         x_recon_denorm_flat = X_test_pca[i, :latent_dimensions] @ pca.components_[:latent_dimensions, :] + pca.mean_
#         x_patient, x_recon_denorm = x_patient_flat.reshape(128,128,32), x_recon_denorm_flat.reshape(128,128,32)
#         metrics = aef.reconstruction_metrics(x_patient, x_recon_denorm, n_training, splitname, latent_dimensions, n_training+1+i, savemetrics=False, analysis="PCA")
#         all_metrics.append(metrics)
#     aef.ae_aggregate_metrics(all_metrics, n_training, splitname, latent_dimensions, analysis="PCA")

# Compare selected patients 
x_recon_denorm_flat_j = X_test_pca[:, :latent_dimensions][patient_toplot-n_training-1] @ pca.components_[:latent_dimensions, :] + np.mean(X_train, axis=0, keepdims=True)[0]
x_recon_denorm_j = x_recon_denorm_flat_j.reshape(128,128,32)
aef.ae_plotcompare_onepatient(x_recon_denorm_j, patient_toplot, "registered_framesBIS", splitname, latent_dimensions,  details_rec = "PCArec")
