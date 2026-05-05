# src/training/ae_training.py
"""
Training, dataset preparation, and metrics aggregation for the 3D autoencoder.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER
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
    imagesource="registered_frames",
    vectorsource="X_vectors",
    recalculateXvector=False,
    image_roi_only=True,
    n_jobs=1,
    validation=False,
    n_validation=24,
    use_both_frames=False,
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
 
    If use_both_frames=True:
        Loads ED and ES frames separately, splits each by patient index,
        then concatenates per split — guaranteeing both frames of a patient
        are always in the same split (no data leakage).
    """
    def _load_X(frame_type,):
        return pcs.get_vectorsarray(
            source_folder=imagesource,
            pca_folder=vectorsource,
            recalculate=recalculateXvector,
            image_roi_only=image_roi_only,
            flatten=False,
            n_jobs=n_jobs,
            frame_type=frame_type,
        )
 
    X_ED = _load_X(frame_type="ED")
 
    if use_both_frames:
        X_ES = _load_X(frame_type="ES")
        if X_ED.shape != X_ES.shape:
            raise ValueError(
                f"Shape mismatch between ED {X_ED.shape} and ES {X_ES.shape}"
            )
 
    # Normalize on ED development pool only (stable reference)
    X_maxnorm = np.percentile(X_ED[:n_patients], percentile_max)
 
    def _normalize(X):
        return np.clip(X, 0, X_maxnorm) / X_maxnorm
 
    X_ED = _normalize(X_ED)
    if use_both_frames:
        X_ES = _normalize(X_ES)
 
    def to_tensor_dataset(X_sub):
        X_sub = np.transpose(X_sub, (0, 3, 1, 2))    # -> (N, 32, 128, 128)
        X_sub = X_sub[:, np.newaxis, :, :, :]         # -> (N, 1, 32, 128, 128)
        X_sub = X_sub.astype(np.float32, copy=False)
        return TensorDataset(torch.from_numpy(X_sub).float())
 
    def _split_and_combine(start, end):
        sub_ED = X_ED[start:end]
        if not use_both_frames:
            return sub_ED
        sub_ES = X_ES[start:end]
        return np.concatenate([sub_ED, sub_ES], axis=0)
 
    if validation:
        n_train = n_patients - n_validation
        if n_train <= 0:
            raise ValueError("n_patients - n_validation must be > 0")
 
        train_dataset      = to_tensor_dataset(_split_and_combine(0, n_train))
        validation_dataset = to_tensor_dataset(_split_and_combine(n_train, n_patients))
        test_dataset       = to_tensor_dataset(_split_and_combine(n_patients, None))
 
    else:
        train_dataset      = to_tensor_dataset(_split_and_combine(0, n_patients))
        validation_dataset = None
        test_dataset       = to_tensor_dataset(_split_and_combine(n_patients, None))
 
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

def ae_aggregate_metrics(all_metrics, simulation_name, n_epochs, metrics_dataset, experiment_name="baseline", ae = True, save_summary=True):
    """
    Aggregate reconstruction metrics over all patients.
    Saves in TEMPODATA_FOLDER/autoencoder/{simulation_name}/{experiment_name}/_{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")

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
        # Build filenames
        if n_epochs is None:
            local_filename = f"_summarymetrics_{metrics_dataset}.txt"
            results_filename = f"{simulation_name}_{experiment_name}_summarymetrics_{metrics_dataset}.txt"
        else:
            local_filename = f"_{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt"
            results_filename = f"{simulation_name}_{experiment_name}_{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt"

        def _write_summary(path):
            with open(path, "w") as f:
                for metric_name, stats in summary.items():
                    f.write(f"{metric_name}\n")
                    for stat_name, value in stats.items():
                        f.write(f"  {stat_name}: {value}\n")
                    f.write("\n")

        # Save alongside model files
        if ae:
            local_path = TEMPODATA_FOLDER / "autoencoder" / simulation_name / experiment_name / local_filename
        else :
            local_path = TEMPODATA_FOLDER / "pca_allpatients_res" / simulation_name  / f"{experiment_name}{local_filename}"
        _write_summary(local_path)

    return summary


def _compute_validation_loss(model, validation_dataset, batch_size, device, criterion):
    """
    Compute mean reconstruction loss on the validation set.
    No gradient computation — fast forward pass only.
    """
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )
 
    model.eval()
    total_loss = 0.0
 
    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            x_recon, _ = model(x_batch)
            loss = criterion(x_recon, x_batch)
            total_loss += loss.item()
 
    return total_loss / len(val_loader)
 
 
def ae_training_early_stopping(
    train_dataset,
    validation_dataset,
    simulation_name,
    model_name,
    latent_dimensions,
    n_epochs=300,
    batch_size=1,
    lr=1e-3,
    patience=20,
    patience_scheduler=8,      
    recalculateAE=True,
    load_epoch=None,
    experiment_name="baseline",  
):
    """
    Train the 3D autoencoder with early stopping based on validation loss.
 
    At every epoch:
      - training loss is computed (free, already done during training)
      - validation loss is computed (cheap forward pass, no gradients)
 
    The best model (lowest validation loss) is saved during training.
    Training stops early if validation loss does not improve for `patience` epochs.
 
    Parameters
    ----------
    train_dataset : TensorDataset
    validation_dataset : TensorDataset
    simulation_name : str
        Base name, e.g. "AE3dConv_96patients_split0_4dims"
    model_name : str
    latent_dimensions : int
    n_epochs : int
        Maximum number of training epochs.
    batch_size : int
    lr : float
    patience : int
        Number of epochs without validation improvement before stopping.
    recalculateAE : bool
        If False, load an existing best model instead of training.
    load_epoch : int or None
        Required when recalculateAE=False. Specifies which saved model to load,
        e.g. load_epoch=42 loads "_best_42epochs.pth".
 
    Returns
    -------
    model : nn.Module
        Best model, loaded and set to eval mode.
    best_epoch : int
        Epoch at which the best validation loss was reached.
    loss_history : dict
        {"train": [float, ...], "validation": [float, ...]}
        One value per epoch (up to early stopping or n_epochs).
 
    Saved files
    -----------
    {simulation_name}/_best_{best_epoch}epochs.pth
    {simulation_name}/_best_{best_epoch}epochs_loss.txt
    """
    device = get_device()
    output_dir = TEMPODATA_FOLDER / "autoencoder" / simulation_name / experiment_name
    # output_dir = TEMPODATA_FOLDER / "autoencoder" / simulation_name
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Load existing model ───────────────────────────────────────────────────
    if not recalculateAE:
        if load_epoch is None:
            raise ValueError(
                "load_epoch must be specified when recalculateAE=False. "
                "Example: load_epoch=42 loads '_best_42epochs.pth'."
            )
 
        best_path = output_dir / f"_best_{load_epoch}epochs.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"Model file not found: {best_path}")
 
        print(f"Loading existing best model: {best_path}")
        model = build_autoencoder(model_name, latent_dimensions).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
 
        loss_history = {"train": [], "validation": []}
        loss_path = output_dir / f"_best_{load_epoch}epochs_loss.txt"
        if loss_path.exists():
            with open(loss_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("train:"):
                        parts = line.split("validation:")
                        train_val = float(parts[0].replace("train:", "").strip())
                        val_val = float(parts[1].strip())
                        loss_history["train"].append(train_val)
                        loss_history["validation"].append(val_val)
 
        return model, load_epoch, loss_history
 
    # ── Training ─────────────────────────────────────────────────────────────
    model = build_autoencoder(model_name, latent_dimensions).to(device)
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )
 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience_scheduler,
        )

    loss_history = {"train": [], "validation": []}
 
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    temp_best_path = output_dir / "_best_model_temp.pth"
 
    print(
        f"Training {model_name} | device={device} | "
        f"batch_size={batch_size} | patience={patience} | max_epochs={n_epochs}"
    )
 
    for epoch in range(n_epochs):
 
        # Training pass
        model.train()
        epoch_train_loss = 0.0
 
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad()
            x_recon, _ = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
 
        avg_train_loss = epoch_train_loss / len(train_loader)
        loss_history["train"].append(avg_train_loss)
 
        # Validation pass
        avg_val_loss = _compute_validation_loss(
            model, validation_dataset, batch_size, device, criterion
        )
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        loss_history["validation"].append(avg_val_loss)
 
        current_epoch = epoch + 1
        is_best = avg_val_loss < best_val_loss
        print(
            f"Epoch {current_epoch}/{n_epochs} "
            f"| train: {avg_train_loss:.6f} "
            f"| val: {avg_val_loss:.6f}"
            f"| lr: {current_lr:.2e}"
            + (" ✓ best" if is_best else f" (no improvement for {epochs_without_improvement + 1} epochs)")
        )
 
        # Save best model
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = current_epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), temp_best_path)
        else:
            epochs_without_improvement += 1
 
        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered at epoch {current_epoch}. "
                f"Best epoch: {best_epoch} (val loss: {best_val_loss:.6f})."
            )
            break
 
    # Rename temp file to permanent name
    final_best_path = output_dir / f"_best_{best_epoch}epochs.pth"
    temp_best_path.rename(final_best_path)
    print(f"Best model saved: {final_best_path}")
 
    # Save loss history
    loss_path = output_dir / f"_best_{best_epoch}epochs_loss.txt"
    with open(loss_path, "w") as f:
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write("\n")
        for tl, vl in zip(loss_history["train"], loss_history["validation"]):
            f.write(f"train: {tl}  validation: {vl}\n")
 
    # Load and return best model
    model.load_state_dict(torch.load(final_best_path, map_location=device))
    model.eval()
 
    return model, best_epoch, loss_history