# src/training/ae_training.py
"""
Training, dataset preparation, and metrics aggregation for the 3D autoencoder.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.config import TEMPODATA_FOLDER
from src.models import pca_spatial as pcs
from src.models.ae_models import build_autoencoder


def get_device(verbose=False):
    """
    Return the best available torch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    return device


def ae_build_basename(model_name, n_patients, splitname, latent_dim, n_epochs=None):
    """
    Build a consistent base filename for AE experiments.

    Example without epochs:
        AE3dFCDeep_96patients_split0_20dims

    Example with epochs:
        AE3dFCDeep_96patients_split0_20dims_50epochs
    """
    parts = [
        model_name,
        f"{n_patients}patients",
        splitname,
        f"{latent_dim}dims",
    ]

    if n_epochs is not None:
        parts.append(f"{n_epochs}epochs")

    return "_".join(parts)


def ae_getdataset(
    n_patients,
    percentile_max=99.9,
    imagesource="registered_framesBIS",
    vectorsource="X_vectors",
    recalculateXvector=False,
    image_roi_only=True,
    details_str="REGvoxROI",
    n_jobs=1,
    validation=False,
    n_validation=24
):
    """
    Build train / validation / test datasets.

    If validation=False:
        train = patients 1..n_patients
        validation = None
        test = remaining patients

    If validation=True:
        train = first (n_patients - n_validation) patients
        validation = next n_validation patients
        test = remaining patients
    """
    X_ini = pcs.get_vectorsarray(
        imagesource,
        vectorsource,
        recalculate=recalculateXvector,
        image_roi_only=image_roi_only,
        details_str=details_str,
        flatten=False,
        n_jobs=n_jobs
    )

    # Normalize on full development pool only
    X_maxnorm = np.percentile(X_ini[:n_patients], percentile_max)

    X_np = np.clip(X_ini, 0, X_maxnorm)
    X_np = X_np / X_maxnorm

    def to_tensor_dataset(X_sub):
        X_sub = np.transpose(X_sub, (0, 3, 1, 2))   # -> (N, 32, 128, 128)
        X_sub = X_sub[:, np.newaxis, :, :, :]       # -> (N, 1, 32, 128, 128)
        X_sub = X_sub.astype(np.float32, copy=False)
        return TensorDataset(torch.from_numpy(X_sub).float())

    if validation:
        n_train = n_patients - n_validation
        if n_train <= 0:
            raise ValueError("n_patients - n_validation must be > 0")

        X_train = X_np[:n_train]
        X_validation = X_np[n_train:n_patients]
        X_test = X_np[n_patients:]

        train_dataset = to_tensor_dataset(X_train)
        validation_dataset = to_tensor_dataset(X_validation)
        test_dataset = to_tensor_dataset(X_test)

    else:
        X_train = X_np[:n_patients]
        X_test = X_np[n_patients:]

        train_dataset = to_tensor_dataset(X_train)
        validation_dataset = None
        test_dataset = to_tensor_dataset(X_test)

    return train_dataset, validation_dataset, test_dataset, X_maxnorm


def dataset_for_metrics(metrics_dataset, train_dataset, validation_dataset, test_dataset, n_train, n_validation=0):
    """
    Return the appropriate dataset and patient offset for metric computation.
    """
    if metrics_dataset == "train":
        dataset = train_dataset
        patient_offset = 0

    elif metrics_dataset == "validation":
        if validation_dataset is None:
            raise ValueError("validation_dataset is None but metrics_dataset='validation'")
        dataset = validation_dataset
        patient_offset = n_train

    elif metrics_dataset == "test":
        dataset = test_dataset
        patient_offset = n_train + n_validation

    else:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")

    return dataset, patient_offset


def ae_training(
    dataset,
    simulation_name,
    model_name,
    latent_dimensions,
    n_epochs=10,
    recalculateAE=True,
    batch_size=1,
    lr=1e-3,
    checkpoint_epochs=None
):
    """
    Train or load the 3D autoencoder.

    Naming:
    - final model: {simulation_name}_{n_epochs}epochs.pth
    - loss file:   {simulation_name}_{n_epochs}epochs_loss.txt

    where simulation_name is now something like:
    AE3dFCDeep_96patients_split0_20dims
    """
    device = get_device()

    final_model_path = (
        TEMPODATA_FOLDER
        / "autoencoder/"
        / simulation_name
        / f"_{n_epochs}epochs.pth"
    )

    loss_path = (
        TEMPODATA_FOLDER
        / "autoencoder/"
        / simulation_name
        / f"_{n_epochs}epochs_loss.txt"
    )

    if checkpoint_epochs is None:
        checkpoint_epochs = []
    checkpoint_epochs = set(checkpoint_epochs)

    if recalculateAE:

        model = build_autoencoder(model_name, latent_dimensions).to(device)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda")
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        epoch_losses = []

        print(f"Training {model_name} on {device} with batch_size={batch_size}")

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for (x_batch,) in loader:
                x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))

                optimizer.zero_grad()
                x_recon, z = model(x_batch)
                loss = criterion(x_recon, x_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

            current_epoch = epoch + 1
            print(f"Epoch {current_epoch}/{n_epochs} - Loss: {avg_loss:.6f}")

            if current_epoch in checkpoint_epochs:
                checkpoint_path = (
                    TEMPODATA_FOLDER
                    / "autoencoder/"
                    / simulation_name
                    / f"_{current_epoch}epochs.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        torch.save(model.state_dict(), final_model_path)

        # Save loss history
        with open(loss_path, "w") as f:
            for i, loss in enumerate(epoch_losses):
                f.write(f"Epoch {i+1}: {loss}\n")

        model.eval()

    else:
        model = build_autoencoder(model_name, latent_dimensions).to(device)
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()

    return model


def ae_reconstructX(patient_tensor, X_maxnorm, model):
    """
    Reconstruct one patient volume with the trained model.
    """
    device = next(model.parameters()).device
    x_patient = patient_tensor[0].unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        x_recon, z = model(x_patient)

    # move back to CPU for numpy conversion
    x_patient_3d = x_patient.squeeze(0).squeeze(0).detach().cpu().numpy()   # (32, 128, 128)
    x_patient_3d = np.transpose(x_patient_3d, (1, 2, 0))                    # (128, 128, 32)
    x_ini_denorm = x_patient_3d * X_maxnorm

    x_recon_np = x_recon.squeeze(0).squeeze(0).detach().cpu().numpy()       # (32, 128, 128)
    x_recon_np = np.transpose(x_recon_np, (1, 2, 0))                        # (128, 128, 32)
    x_recon_denorm = x_recon_np * X_maxnorm

    return x_ini_denorm, x_recon_denorm


def reconstruction_metrics(x_true, x_pred, patient_number, simulation_name, n_epochs, metrics_dataset, savemetrics=True):
    """
    Compute reconstruction metrics between two 3D images of identical shape.
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train' or 'test'")

    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch: {x_true.shape} vs {x_pred.shape}")

    diff = x_true - x_pred

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "patient_number": int(patient_number)
    }

    if savemetrics:
        if n_epochs is None:
            save_path = (
                TEMPODATA_FOLDER
                / "autoencoder/"
                / simulation_name
                / f"_resultspatient{patient_number}_{metrics_dataset}.txt"
            )
        else:
            save_path = (
                TEMPODATA_FOLDER
                / "autoencoder/"
                / simulation_name
                / f"_{n_epochs}epochs_resultspatient{patient_number}_{metrics_dataset}.txt"
            )

        with open(save_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    return metrics


def ae_aggregate_metrics(all_metrics, simulation_name, n_epochs, metrics_dataset, save_summary=True):
    """
    Aggregate reconstruction metrics over all patients.
    Save either:
    - *_summarymetrics_test.txt
    - *_summarymetrics_train.txt
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train' or 'test'")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]

    summary = {}

    for metric_name in metric_names:
        values = np.array([m[metric_name] for m in all_metrics], dtype=np.float32)

        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    if save_summary:
        if n_epochs is None:
            save_path = (
                TEMPODATA_FOLDER
                / "autoencoder/"
                / simulation_name
                / f"_summarymetrics_{metrics_dataset}.txt"
            )
        else:
            save_path = (
                TEMPODATA_FOLDER
                / "autoencoder/"
                / simulation_name
                / f"_{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt"
            )

        with open(save_path, "w") as f:
            for metric_name, stats in summary.items():
                f.write(f"{metric_name}\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value}\n")
                f.write("\n")

    return summary


def ae_select_representative_patients(all_metrics):
    """
    Select:
    - patient with lowest R2
    - patient with highest R2
    - patient with R2 closest to mean R2
    """
    r2_values = np.array([m["R2"] for m in all_metrics], dtype=np.float32)
    patient_numbers = np.array([m["patient_number"] for m in all_metrics], dtype=int)

    mean_r2 = np.mean(r2_values)

    idx_min = np.argmin(r2_values)
    idx_max = np.argmax(r2_values)
    idx_mean = np.argmin(np.abs(r2_values - mean_r2))

    selected = {
        "worst": {
            "patient_number": int(patient_numbers[idx_min]),
            "R2": float(r2_values[idx_min]),
        },
        "best": {
            "patient_number": int(patient_numbers[idx_max]),
            "R2": float(r2_values[idx_max]),
        },
        "closest_to_mean": {
            "patient_number": int(patient_numbers[idx_mean]),
            "R2": float(r2_values[idx_mean]),
        },
    }

    return selected