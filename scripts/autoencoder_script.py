# scripts/autoencoder_script.py
"""
Script for the entoencoder model to compare patients hearts
"""

import torch 

from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER
from src.training import ae_training as aet
from src.visualization import ae_plots as aep

# User choices : DATA
n_development = 120
validation = True
n_validation = 24 if validation else 0
splitname = "split0"
# User choices : MODEL
model_name = "AE3dConv" # "AE3dCurrent", or "AE3dFCDeep" , "AE3dConv"
latent_dimensions = 4
latdim_list = [4, 20, 80] # latdim_list = [1,2,3,4,5,7,10,15,20,30,40,50,80,119] or latdim_list = [4, 8, 12, 16, 20, 32, 40, 60, 80, 120] for AE3dConv 
n_epochs = 100 
checkpoint_epochs = [20, 50]
# User choices: METRICS 
metrics_dataset = "validation" # "test,  "validation", or "train"

# Get data
n_train_effective = n_development - n_validation if validation else n_development
train_dataset, validation_dataset, test_dataset, X_maxnorm = aet.ae_getdataset(
    n_patients=n_development,
    validation=validation,
    n_validation=n_validation,
    imagesource="registered_framesBIS"
)


#     for latent_dimensions in latdim_list:
simulation_name = f"{model_name}_{n_train_effective}patients_{splitname}_{latent_dimensions}dims"
for n_epochs in sorted(set((checkpoint_epochs or []) + [n_epochs])):
    # Model training/loading 
    model = aet.ae_training(
        train_dataset,
        simulation_name=simulation_name,
        model_name=model_name,
        latent_dimensions=latent_dimensions,
        n_epochs=n_epochs,
        batch_size=1,
        recalculateAE=False,
        checkpoint_epochs=checkpoint_epochs
    )

    for metrics_dataset in ["validation", "train"]:
        #   Get efficienty metrics for every test patient and agregate
        dataset_for_eval, patient_offset = aet.dataset_for_metrics(
            metrics_dataset,
            train_dataset,
            validation_dataset,
            test_dataset,
            n_train=n_train_effective,
            n_validation=n_validation
        )
        all_metrics = []
        for (i,patient_tensor) in enumerate(dataset_for_eval):
            x_patient, x_recon_denorm = aet.ae_reconstructX(patient_tensor, X_maxnorm, model)
            metrics = aet.reconstruction_metrics(
                x_true=x_patient,
                x_pred=x_recon_denorm,
                patient_number=patient_offset + 1 + i,
                simulation_name=simulation_name,
                n_epochs = n_epochs,
                metrics_dataset=metrics_dataset,
                savemetrics=False
            )
            all_metrics.append(metrics)
        aet.ae_aggregate_metrics(all_metrics, simulation_name=simulation_name, n_epochs = n_epochs, metrics_dataset=metrics_dataset)
        aep.plot_ae_loss_from_txt(TEMPODATA_FOLDER / "autoencoder/AE3d_120patients_split0_latent119_200epochs_loss.txt", save_path=RESULTS_FOLDER / "AE3d_120patients_split0_latent119_lossepochplot.png")

#  Plot a few patients
#  ADD A FUNCTION TO SAVE AND LOAD ALL_METRICS (LATER)
selected_patients = aet.ae_select_representative_patients(all_metrics)
patient_numbers = [v['patient_number'] for v in selected_patients.values()] # patient_numbers = [134, 136, 144]
aep.ae_plotcompare_selected(patient_numbers, n_training, train_dataset, test_dataset, X_maxnorm, model, splitname, latent_dimensions, n_epochs, metrics_dataset=metrics_dataset, validation_dataset=validation_dataset)

# #  Plot AE VS PCA results 
# aep.plot_summarymetrics_vs_latentdim(TEMPODATA_FOLDER / "autoencoder", splitname=splitname, n_patients = n_training, device_tag = None, batch_size=None, metrics_dataset=metrics_dataset, band_mode=None)
# aep.plot_r2_test_vs_train(
#     TEMPODATA_FOLDER / "autoencoder",
#     splitname=splitname,
#     n_patients=n_training,
#     device_tag=None,
#     batch_size=None,
#     annotate_dims=True
# )

#  # Compare AE models 
# epoch_list = sorted(set((checkpoint_epochs or []) + [n_epochs]))
# figs_by_arch = aep.plot_ae_metrics_vs_latentdim_by_architecture( # 1) One figure per architecture, 3 curves = epochs
#     results_folder=TEMPODATA_FOLDER / "autoencoder",
#     splitname=splitname,
#     n_patients=n_train_effective,
#     metrics_dataset=metrics_dataset,
#     model_names=["AE3dCurrent", "AE3dFCDeep", "AE3dConv"],
#     epoch_list=epoch_list,
#     xscale="log"
# )
# figs_by_epoch = aep.plot_ae_metrics_vs_latentdim_by_epoch( # 2) One figure per epoch, 3 curves = architectures
#     results_folder=TEMPODATA_FOLDER / "autoencoder",
#     splitname=splitname,
#     n_patients=n_train_effective,
#     metrics_dataset=metrics_dataset,
#     model_names=["AE3dCurrent", "AE3dFCDeep", "AE3dConv"],
#     epoch_list=epoch_list,
#     xscale="log"
# )
# fig_scatter, ax_scatter, paired_records = aep.plot_ae_r2_validation_vs_train_scatter( # 3) Scatter train R2 vs validation R2
#     results_folder=TEMPODATA_FOLDER / "autoencoder",
#     splitname=splitname,
#     n_patients=n_train_effective,
#     model_names=["AE3dCurrent", "AE3dFCDeep", "AE3dConv"],
#     epoch_list=epoch_list,
#     annotate_dims=True
# )