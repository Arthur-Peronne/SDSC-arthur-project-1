# scripts/run_autoencoder.py
"""
Script for training and evaluating the 3D autoencoder on cardiac MRI data.
Uses early stopping based on validation loss.
"""

from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER
from src.training import ae_training as aet
from src.visualization import ae_plots as aep

# ── User choices : DATA ───────────────────────────────────────────────────────
use_both_frames = True
n_development = 120
n_validation = 20
splitname = "split0"
recalculateX = False

# ── User choices : MODEL ──────────────────────────────────────────────────────
multiple_modelsanddims = False 
model_name = "AE3dFCDeep"         # "AE3dCurrent", "AE3dFCDeep", "AE3dConv", "AE3dLinear"
latent_dimensions = 120  # Among [4, 8, 12, 20, 28, 40, 60, 80, 100] 

# ── User choices : TRAINING ───────────────────────────────────────────────────
recalculateAE = False
load_epoch = 71               # required if recalculateAE=False, e.g. load_epoch=42
experiment_name = "baseline"    # "baseline" or other
n_epochs = 500                  # maximum epochs (early stopping will likely trigger before), baseline 500
patience = 20                   # baseline: 20
patience_scheduler = 8         # baseline : 8
batch_size = 1                  # baseline: 1
lr = 1e-5                      # baseline: 1e-5, 1e-6 for Linear

# ── User choices : RECONSTRUCTION ───────────────────────────────────────────────────
plot_reconstruction = True 
recons_auto = True            # Automatically choose 3 patients good/medium/bad reconstruct
patients_torecons_manual = [(30,"ES"), (110, "ED"),(130, "ED")] # else manual choice

# ── Load data ─────────────────────────────────────────────────────────────────
n_train_effective = n_development - n_validation
n_train_images = n_train_effective * 2 if use_both_frames else n_train_effective
n_val_images = n_validation * 2 if use_both_frames else n_validation
# n_test_images = (150-n_development) * 2 if use_both_frames else (150-n_development) 

train_dataset, validation_dataset, test_dataset, X_maxnorm = aet.ae_getdataset(
    n_patients=n_development,
    validation=True,
    n_validation=n_validation,
    imagesource="registered_frames", 
    use_both_frames=use_both_frames,
    recalculateXvector=recalculateX,
)

# ── Train (or reload) ─────────────────────────────────────────────────────────
if not multiple_modelsanddims:
    simulation_name = f"{model_name}_{n_train_images}patients_{splitname}_{latent_dimensions}dims"
    model, best_epoch, loss_history = aet.ae_training_early_stopping(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        simulation_name=simulation_name,
        model_name=model_name,
        latent_dimensions=latent_dimensions,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        patience_scheduler=patience_scheduler,
        recalculateAE=recalculateAE,
        load_epoch=load_epoch,
        experiment_name=experiment_name,
    )
        # ── Plot train / validation loss curves ───────────────────────────────────────
    aep.plot_train_val_loss(
        loss_history=loss_history,
        best_epoch=best_epoch,
        simulation_name=simulation_name,
        experiment_name=experiment_name,
    )

    # ── Compute R² on train and validation sets ───────────────────────────────────
    for metrics_dataset in ["train", "validation", "test"]:
        dataset_for_eval, patient_offset = aet.dataset_for_metrics(
            metrics_dataset,
            train_dataset,
            validation_dataset,
            test_dataset,
            n_train=n_train_images,
            n_validation=n_val_images,
        )

        all_metrics = []
        for i, patient_tensor in enumerate(dataset_for_eval):
            x_patient, x_recon_denorm = aet.ae_reconstructX(patient_tensor, X_maxnorm, model)
            metrics = aet.reconstruction_metrics(
                x_true=x_patient,
                x_pred=x_recon_denorm,
                patient_number=patient_offset + 1 + i,
                simulation_name=simulation_name,
                n_epochs=best_epoch,
                metrics_dataset=metrics_dataset,
                savemetrics=False,
            )
            all_metrics.append(metrics)

        aet.ae_aggregate_metrics(
            all_metrics,
            simulation_name=simulation_name,
            n_epochs=best_epoch,
            metrics_dataset=metrics_dataset,
            experiment_name=experiment_name,
        )
else:
    for latent_dimensions in [200]:  # [4, 8, 12, 20, 28, 40, 60, 88, 120, 160, 200]:
        for model_name in ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"]:
            simulation_name = f"{model_name}_{n_train_images}patients_{splitname}_{latent_dimensions}dims"

            model, best_epoch, loss_history = aet.ae_training_early_stopping(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                simulation_name=simulation_name,
                model_name=model_name,
                latent_dimensions=latent_dimensions,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                patience_scheduler=patience_scheduler,
                recalculateAE=recalculateAE,
                load_epoch=load_epoch,
                experiment_name=experiment_name,
            )
        # ── Plot train / validation loss curves ───────────────────────────────────────
            aep.plot_train_val_loss(
                loss_history=loss_history,
                best_epoch=best_epoch,
                simulation_name=simulation_name,
                experiment_name=experiment_name,
            )

            # ── Compute R² on train and validation sets ───────────────────────────────────
            for metrics_dataset in ["train", "validation", "test"]:
                dataset_for_eval, patient_offset = aet.dataset_for_metrics(
                    metrics_dataset,
                    train_dataset,
                    validation_dataset,
                    test_dataset,
                    n_train=n_train_images,
                    n_validation=n_val_images,
                )

                all_metrics = []
                for i, patient_tensor in enumerate(dataset_for_eval):
                    x_patient, x_recon_denorm = aet.ae_reconstructX(patient_tensor, X_maxnorm, model)
                    metrics = aet.reconstruction_metrics(
                        x_true=x_patient,
                        x_pred=x_recon_denorm,
                        patient_number=patient_offset + 1 + i,
                        simulation_name=simulation_name,
                        n_epochs=best_epoch,
                        metrics_dataset=metrics_dataset,
                        savemetrics=False,
                    )
                    all_metrics.append(metrics)

                aet.ae_aggregate_metrics(
                    all_metrics,
                    simulation_name=simulation_name,
                    n_epochs=best_epoch,
                    metrics_dataset=metrics_dataset,
                    experiment_name=experiment_name,
                )

# ── Plot reconstruction for representative patients ───────────────────────────
if plot_reconstruction:
    if recons_auto:
        selected = aep.ae_select_representative_patients(
            all_metrics,
            use_both_frames=use_both_frames,
            n_train_images=n_train_images,
            n_val_images=n_val_images,
            n_development=n_development,
        )
        patients_torecons = [(v["real_patient"], v["frame_type"]) for v in selected.values()]
    else:
        patients_torecons = patients_torecons_manual

    aep.ae_plotcompare_selected(
        patients_torecons=patients_torecons,
        use_both_frames=use_both_frames,
        n_development=n_development,
        n_train_images=n_train_images,
        n_val_images=n_val_images,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        X_maxnorm=X_maxnorm,
        model=model,
        model_name=model_name,
        split_name=splitname,
        latent_dimensions=latent_dimensions,
        n_epochs=best_epoch,
        validation_dataset=validation_dataset,
    )



# ── Commented: loop over latent dims ──────────────────────────────────────────
# for latent_dimensions in latdim_list:
#     simulation_name = f"{model_name}_{n_train_effective}patients_{splitname}_{latent_dimensions}dims"
#     model, best_epoch, loss_history = aet.ae_training_early_stopping(...)
#     ...

# ── Commented: comparison plots (AE vs PCA, architectures, scatter) ───────────
# aep.plot_summarymetrics_vs_latentdim(...)
# aep.plot_ae_metrics_vs_latentdim_by_architecture(...)
# aep.plot_ae_metrics_vs_latentdim_by_epoch(...)
# aep.plot_ae_r2_validation_vs_train_scatter(...)

















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
