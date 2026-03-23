# src/autoencoder_script.py
"""
Script for the entoencoder model to compare patients hearts
"""

from paths import *
import autoencoder_functions as aef 

# User choices : DATA
n_training = 120
splitname = "split0"
# User choices : MODEL
latent_dimensions = 119
latdim_list = [1,2,3,4,5,7,10,15,20,30,40,50,80,119]
n_epochs = 50 

# # Get data
# train_dataset, test_dataset, X_maxnorm = aef.ae_getdataset(n_training, imagesource = "registered_framesBIS") # Get data

# # for latent_dimensions in latdim_list:
# # Model training/loading 
# model = aef.ae_training(train_dataset, latent_dimensions, splitname = splitname, n_epochs = n_epochs, recalculateAE=False) # Model calc/load

# # Get efficienty metrics for every test patient and agregate
# all_metrics = []
# for (i,patient_tensor) in enumerate(test_dataset):
#     x_patient, x_recon_denorm = aef.ae_reconstructX(patient_tensor, X_maxnorm, model)
#     metrics = aef.reconstruction_metrics(x_patient, x_recon_denorm, n_training, splitname, latent_dimensions, n_epochs, n_training+1+i, savemetrics=False)
#     all_metrics.append(metrics)
# aef.ae_aggregate_metrics(all_metrics, n_training, splitname, latent_dimensions, n_epochs)

# Plot a few patients
# ADD A FUNCTION TO SAVE AND LOAD ALL_METRICS (LATER)
# selected_patients = aef.ae_select_representative_patients(all_metrics)
# patient_numbers = [v['patient_number'] for v in selected_patients.values()] # patient_numbers = [134, 136, 144]
# aef.ae_plotcompare_selected(patient_numbers, n_training, test_dataset, X_maxnorm, model, splitname, latent_dimensions, n_epochs)

# Plot AE VS PCA results 
aef.plot_summarymetrics_vs_latentdim(path_tempodata_folder + "autoencoder", splitname="split0", n_patients=n_training, ae_epochs=50)
# aef.plot_ae_loss_from_txt(path_tempodata_folder + "autoencoder/AE3d_120patients_split0_latent119_200epochs_loss.txt", save_path = path_resultsfolder+"AE3d_120patients_split0_latent119_lossepochplot.png")
