# src/pca_vsautoencoder.py
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
images_folder = "registered_framesBIS"
n_training = 120
splitname = "split0"
# User choices : MODEL
latdim_list = [1,2,3,4,5,7,10,15,20,30,40,50,80,119]
# User choices : METRICS 
metrics_dataset = "train"   # "test" or "train"
# User choices : RECONSTRUCTION 
patient_toplot = 144
latent_dim_plot = 119

# Get data
X = pef.get_vectorsarray(images_folder, "X_vectors", details_str = "REGvoxROI", image_roi_only=True, recalculate= False)
X_train, X_test = X[:n_training], X[n_training:]
del X 
dataset_for_metrics, patient_offset = aef.dataset_for_metrics(metrics_dataset, X_train, X_test, n_training)

# Model training/loading 
pca_filename = "PCA_" + repr(n_training) + "patients_" + splitname
pca, X_train_pca, meta = pef.pca_patients(X_train, "autoencoder", pca_filename,  normalize_rows=False, recalculatePCA = False) # PCA without normalization of brightness accross patients (IMPORTANT, following reconstruction wrong if normalize_rows is not False here)
X_test_pca = pca.transform(X_test)
X_metrics_pca, _ = aef.dataset_for_metrics(metrics_dataset, X_train_pca, X_test_pca, n_training)


# Reconstruct test patient from PCA and calculate results averaged over test sample
for latent_dimensions in latdim_list:
    simulation_name = f"PCA_{n_training}patients_{splitname}_{latent_dimensions}dims"
    all_metrics = []
    for (i, x_patient_flat) in enumerate(dataset_for_metrics):
        x_recon_denorm_flat = X_metrics_pca[i, :latent_dimensions] @ pca.components_[:latent_dimensions, :] + pca.mean_ # Calculate x_reconstructed from PCA
        x_patient, x_recon_denorm = x_patient_flat.reshape(128,128,32), x_recon_denorm_flat.reshape(128,128,32) # Put x_reconstructed and x_original back in 3D shape
        # metrics = aef.reconstruction_metrics(x_patient, x_recon_denorm, n_training, splitname, latent_dimensions, n_training+1+i, n_epochs=None, savemetrics=False, analysis="PCA")
        metrics = aef.reconstruction_metrics(
            x_true=x_patient,
            x_pred=x_recon_denorm,
            patient_number=patient_offset + 1 + i,
            simulation_name=simulation_name,
            n_epochs=None,
            metrics_dataset=metrics_dataset,
            savemetrics=False
        )
        all_metrics.append(metrics)
    aef.ae_aggregate_metrics(all_metrics, simulation_name=simulation_name, n_epochs=None, metrics_dataset=metrics_dataset)

# Compare selected patients 
# Patient in test or train?
if patient_toplot > n_training:
    idx_plot = patient_toplot - n_training - 1
    x_plot_pca = X_test_pca
else:
    idx_plot = patient_toplot - 1
    x_plot_pca = X_train_pca
# Plot
x_recon_denorm_flat_j = x_plot_pca[idx_plot, :latent_dim_plot] @ pca.components_[:latent_dim_plot, :] + pca.mean_
x_recon_denorm_j = x_recon_denorm_flat_j.reshape(128,128,32)
aef.ae_plotcompare_onepatient(x_recon_denorm_j, patient_toplot, "registered_framesBIS", splitname, latent_dim_plot,  details_rec = "PCArec")
