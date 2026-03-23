# src/autoencoder_functions.py
"""
Functions for the script for the entoencoder model to compare patients hearts
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import nibabel as nib
import os
import re
import glob
import matplotlib.pyplot as plt

from paths import * 
import pca_eachpatient_functions as pef
import importdata_functions as idf
import visualizeMRI_functions as vmf 

class Conv3DBlock(nn.Module):
    """
    3D convolutional block:
    Conv3D -> InstanceNorm3D -> ReLU -> Conv3D -> InstanceNorm3D -> ReLU
    Optionally followed by MaxPool3D for downsampling.
    """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        if self.downsample:
            x = self.pool(x)

        return x

class UpConv3DBlock(nn.Module):
    """
    3D decoder block:
    ConvTranspose3D (upsampling) -> Conv3D -> InstanceNorm3D -> ReLU -> Conv3D -> InstanceNorm3D -> ReLU
    No skip connections.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv1 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x

class AutoEncoder3D(nn.Module):
    """
    3D convolutional autoencoder for input shape:
        (B, 1, 32, 128, 128)

    Encoder spatial progression:
        (1, 32, 128, 128)
        -> (8, 16, 64, 64)
        -> (16, 8, 32, 32)
        -> (32, 4, 16, 16)
        -> bottleneck conv block -> (64, 4, 16, 16)

    Then:
        flatten -> latent vector z (dim = latent_dim)
        z -> linear -> reshape -> decoder

    Recommended latent_dim:
        10, 20, or 100
    """

    def __init__(self, latent_dim=10, input_shape=(1, 32, 128, 128)):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape  # (C, D, H, W)

        # -----------------
        # Encoder
        # -----------------
        self.enc1 = Conv3DBlock(in_channels=1, out_channels=8, downsample=True)
        self.enc2 = Conv3DBlock(in_channels=8, out_channels=16, downsample=True)
        self.enc3 = Conv3DBlock(in_channels=16, out_channels=32, downsample=True)

        # Bottleneck conv block without further pooling
        self.bottleneck_conv = Conv3DBlock(in_channels=32, out_channels=64, downsample=False)

        # After 3 pools:
        # D: 32 -> 16 -> 8 -> 4
        # H: 128 -> 64 -> 32 -> 16
        # W: 128 -> 64 -> 32 -> 16
        self.feature_shape = (64, 4, 16, 16)
        flattened_size = 64 * 4 * 16 * 16  # 65536

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # -----------------
        # Decoder
        # -----------------
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.dec1 = UpConv3DBlock(in_channels=64, out_channels=32)
        self.dec2 = UpConv3DBlock(in_channels=32, out_channels=16)
        self.dec3 = UpConv3DBlock(in_channels=16, out_channels=8)

        # Final reconstruction layer back to 1 channel
        self.final_conv = nn.Conv3d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Optional final activation:
        # - keep identity if your data are globally normalized but can be negative
        # - use Sigmoid if inputs are scaled to [0, 1]
        # self.final_activation = nn.Identity()
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)              # -> (B, 8, 16, 64, 64)
        x = self.enc2(x)              # -> (B, 16, 8, 32, 32)
        x = self.enc3(x)              # -> (B, 32, 4, 16, 16)
        x = self.bottleneck_conv(x)   # -> (B, 64, 4, 16, 16)

        x = self.flatten(x)           # -> (B, 65536)
        z = self.fc_enc(x)            # -> (B, latent_dim)
        return z

    def decode(self, z):
        x = self.fc_dec(z)            # -> (B, 65536)
        x = x.view(-1, *self.feature_shape)  # -> (B, 64, 4, 16, 16)

        x = self.dec1(x)              # -> (B, 32, 8, 32, 32)
        x = self.dec2(x)              # -> (B, 16, 16, 64, 64)
        x = self.dec3(x)              # -> (B, 8, 32, 128, 128)

        x = self.final_conv(x)        # -> (B, 1, 32, 128, 128)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def reconstruction_metrics(x_true, x_pred, n_patients, splitname, latent_dimensions, n_epochs,  patient_number, savemetrics=True, analysis="AE3d"):
    """
    Compute reconstruction metrics between two 3D images
    of identical shape.
    """
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

    # Save if path is provided
    if savemetrics:
        save_path = path_tempodata_folder + "autoencoder/" + analysis + "_" + repr(n_patients) + "patients_" + splitname + "_latent" + repr(latent_dimensions) + "_" + repr(n_epochs) + "epochs_resultspatient" + repr(patient_number) + ".txt"
        with open(save_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    return metrics

def ae_training(dataset, latent_dimensions, splitname = "", n_epochs = 10,  recalculateAE=True):
    """
    """

    if recalculateAE:   # MODEL FIT 

        model = AutoEncoder3D(latent_dim=latent_dimensions)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        n_epochs = n_epochs

        epoch_losses = []
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for (x_batch,) in loader:
                optimizer.zero_grad()

                x_recon, z = model(x_batch)
                loss = criterion(x_recon, x_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss / len(loader):.6f}")

        outpath = path_tempodata_folder + "autoencoder/AE3d_" + repr(len(dataset)) + "patients_" + splitname + "_latent" + repr(latent_dimensions) + "_" + repr(n_epochs) + "epochs.pth"
        torch.save(model.state_dict(), outpath)
        # Save loss history
        loss_path = outpath.replace(".pth", "_loss.txt")
        with open(loss_path, "w") as f:
            for i, loss in enumerate(epoch_losses):
                f.write(f"Epoch {i+1}: {loss}\n")

    else: # MODEL LOAD 

        model = AutoEncoder3D(latent_dim=latent_dimensions)
        outpath = path_tempodata_folder + "autoencoder/AE3d_" + repr(len(dataset)) + "patients_" + splitname + "_latent" + repr(latent_dimensions) + "_" + repr(n_epochs) + "epochs.pth"
        model.load_state_dict(torch.load(outpath, map_location="cpu"))
        model.eval()

    return model

def ae_getdataset(n_patients, percentile_max = 99.9, imagesource = "registered_framesBIS", vectorsource = "X_vectors", recalculateXvector=False, image_roi_only=True, details_str = "REGvoxROI", n_jobs=1):
    """
    """
    # Load data X
    X_ini = pef.get_vectorsarray(imagesource, vectorsource, recalculate = recalculateXvector, image_roi_only=image_roi_only, details_str = details_str, flatten=False, n_jobs=n_jobs)
    # Normalize global
    X_maxnorm = np.percentile(X_ini, percentile_max)
    X_np = np.clip(X_ini, 0, X_maxnorm)
    X_np = X_np / X_maxnorm
    # Directions in the right order -> TRAIN DATA SET
    X_temptrain = np.transpose(X_np[:n_patients], (0, 3, 1, 2))   # -> (150, 32, 128, 128)
    X_temptrain = X_temptrain[:, np.newaxis, :, :, :]       # -> (150, 1, 32, 128, 128)
    X_temptrain = X_temptrain.astype(np.float32, copy=False)
    X_train = torch.from_numpy(X_temptrain).float()
    # TEST DATASET
    X_temptest = np.transpose(X_np[n_patients:], (0, 3, 1, 2))   # -> (150, 32, 128, 128)
    X_temptest = X_temptest[:, np.newaxis, :, :, :]       # -> (150, 1, 32, 128, 128)
    X_temptest = X_temptest.astype(np.float32, copy=False)
    X_test = torch.from_numpy(X_temptest).float()
    # Transform into Tensor dataset 
    train_dataset, test_dataset = TensorDataset(X_train), TensorDataset(X_test)

    return train_dataset, test_dataset, X_maxnorm


def ae_reconstructX(patient_tensor, X_maxnorm, model):
    """
    """
    # Get patient initial X and its reconstruction from the model
    x_patient = patient_tensor[0].unsqueeze(0) 
    with torch.no_grad():
        x_recon, z = model(x_patient)
    # Put back the right dimensions and de-norm
    x_patient_3d = x_patient.squeeze(0).squeeze(0).cpu().numpy()   # -> (32, 128, 128)
    x_patient_3d = np.transpose(x_patient_3d, (1, 2, 0))           # -> (128, 128, 32)
    x_ini_denorm = x_patient_3d * X_maxnorm 
    x_recon_np = x_recon.squeeze(0).squeeze(0).cpu().numpy()   # -> (32, 128, 128)
    x_recon_np = np.transpose(x_recon_np, (1, 2, 0))           # -> (128, 128, 32)
    x_recon_denorm = x_recon_np * X_maxnorm 
    return x_ini_denorm, x_recon_denorm

def ae_plotcompare_onepatient(x_recon_denorm, patient_number, folder_originals, splitname, latent_dimensions, details_rec = "AErec"):
    """
    """
    # Load initial nii (to get affine, and to plot)
    ini_path = idf.get_patient_modified_path(patient_number, folder_originals)
    ini_nii= nib.load(ini_path)
    reconstructed_nii = nib.Nifti1Image(x_recon_denorm.astype(np.float32), affine=ini_nii.affine, header=ini_nii.header.copy())
    # Modify original to only keep the ROI 
    inimask_path = idf.get_patient_modified_path(patient_number, folder_originals, file_type="mask")
    inimask_nii = nib.load(inimask_path)
    ini_data = ini_nii.get_fdata(dtype=np.float32)
    mask_data = inimask_nii.get_fdata(dtype=np.float32)
    mask_bin = mask_data > 0
    ini_data_roi = ini_data.copy()
    ini_data_roi[~mask_bin] = 0.0
    ini_nii_roi = nib.Nifti1Image(ini_data_roi.astype(np.float32), affine=ini_nii.affine, header=ini_nii.header.copy())
    # Plot
    vmf.plot_oneimg(ini_nii_roi, cmapYN = False, patient_str = "original", file_str = "patient"+repr(patient_number), details_str="REGvoxROI")
    vmf.plot_oneimg(reconstructed_nii, cmapYN = False, cmap = "", patient_str =splitname + "_" + repr(latent_dimensions) + "dims", file_str = "patient"+repr(patient_number), details_str=details_rec)


def ae_aggregate_metrics(all_metrics, n_patients, splitname, latent_dimensions, n_epochs, save_summary=True, analysis = "AE3d"):
    """
    Aggregate reconstruction metrics over all test patients.
    """
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
        save_path = (
            path_tempodata_folder
            + "autoencoder/" + analysis + "_" 
            + repr(n_patients)
            + "patients_"
            + splitname
            + "_latent"
            + repr(latent_dimensions) + "_"
            + repr(n_epochs)
            + "epochs_summarymetrics.txt"
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


def ae_plotcompare_selected(patient_numbers, n_patients, test_dataset, X_maxnorm, model, splitname, latent_dimensions, n_epochs):
    """
    """

    for patient_number in patient_numbers:
        # patient_number = info["patient_number"]

        # recover index inside test_dataset
        test_idx = patient_number - (n_patients + 1)

        patient_tensor = test_dataset[test_idx]
        x_patient, x_recon_denorm = ae_reconstructX(patient_tensor, X_maxnorm, model)

        ae_plotcompare_onepatient(x_recon_denorm, patient_number, "registered_framesBIS", splitname, latent_dimensions, details_rec = "AErec_" + repr(n_epochs) + "trainingepochs")

def _parse_summarymetrics_file(filepath):
    """
    Parse one summary metrics txt file.

    Supported filename patterns:
    - AE3d_<N>patients_<splitname>_latent<D>_<E>epochs_summarymetrics.txt
    - PCA_<N>patients_<splitname>_latent<D>_summarymetrics.txt

    Returns
    -------
    dict
        {
            "analysis": "AE3d" or "PCA",
            "n_patients": int,
            "splitname": str,
            "latent_dim": int,
            "n_epochs": int or None,
            "metrics": {...}
        }
    """
    filename = os.path.basename(filepath)

    # New AE pattern with epoch count
    pattern_ae = r'^(AE3d)_(\d+)patients_(.+?)_latent(\d+)_(\d+)epochs_summarymetrics\.txt$'
    # PCA pattern unchanged
    pattern_pca = r'^(PCA)_(\d+)patients_(.+?)_latent(\d+)_summarymetrics\.txt$'

    match_ae = re.match(pattern_ae, filename)
    match_pca = re.match(pattern_pca, filename)

    if match_ae is not None:
        analysis = match_ae.group(1)
        n_patients = int(match_ae.group(2))
        splitname = match_ae.group(3)
        latent_dim = int(match_ae.group(4))
        n_epochs = int(match_ae.group(5))

    elif match_pca is not None:
        analysis = match_pca.group(1)
        n_patients = int(match_pca.group(2))
        splitname = match_pca.group(3)
        latent_dim = int(match_pca.group(4))
        n_epochs = None

    else:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    metrics = {}
    current_metric = None

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Metric header line, e.g. "MSE"
            if line in {"MSE", "RMSE", "MAE", "R2"}:
                current_metric = line
                metrics[current_metric] = {}
                continue

            # Stat line, e.g. "mean: 123.4"
            if current_metric is not None and ":" in line:
                key, value = line.split(":", 1)
                metrics[current_metric][key.strip()] = float(value.strip())

    return {
        "analysis": analysis,
        "n_patients": n_patients,
        "splitname": splitname,
        "latent_dim": latent_dim,
        "n_epochs": n_epochs,
        "metrics": metrics,
        "filepath": filepath,
    }


def plot_summarymetrics_vs_latentdim(
    results_folder,
    splitname=None,
    n_patients=None,
    # band_mode="std",          # "std", "minmax", or None
    xscale="log",   # NEW
    vline_dim = None,
    ae_epochs=None,
):
    """
    Plot MSE, RMSE, MAE, and R2 vs latent dimension, with AE3d and PCA on the same figure.

    Parameters
    ----------
    results_folder : str
        Folder containing files like:
        - AE3d_120patients_split0_latent1_summarymetrics.txt
        - PCA_120patients_split0_latent1_summarymetrics.txt
    splitname : str or None
        If provided, only files matching this splitname are used.
    n_patients : int or None
        If provided, only files matching this training patient count are used.
    band_mode : {"std", "minmax", None}
        - "std"    -> shaded band mean ± std
        - "minmax" -> shaded band from min to max
        - None     -> no shaded band

    Returns
    -------
    fig, axes, parsed_results
    """
    # if band_mode not in {"std", "minmax", None}:
    #     raise ValueError("band_mode must be one of: 'std', 'minmax', None")

    filepaths = sorted(glob.glob(os.path.join(results_folder, "*_summarymetrics.txt")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No *_summarymetrics.txt files found in: {results_folder}")

    parsed = []
    for fp in filepaths:
        try:
            rec = _parse_summarymetrics_file(fp)
        except ValueError:
            # Ignore files that do not match the naming convention
            continue

        if splitname is not None and rec["splitname"] != splitname:
            continue
        if n_patients is not None and rec["n_patients"] != n_patients:
            continue
        if ae_epochs is not None and rec["analysis"] == "AE3d":
            if rec["n_epochs"] != ae_epochs:
                continue

        parsed.append(rec)

    if len(parsed) == 0:
        raise ValueError("No matching summary metrics files found after filtering.")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]
    analysis_names = ["AE3d", "PCA"]

    # Organize data
    data = {analysis: {metric: [] for metric in metric_names} for analysis in analysis_names}

    for rec in parsed:
        analysis = rec["analysis"]
        latent_dim = rec["latent_dim"]

        if analysis not in data:
            continue

        for metric in metric_names:
            if metric not in rec["metrics"]:
                continue

            entry = {
                "latent_dim": latent_dim,
                "mean": rec["metrics"][metric].get("mean", np.nan),
                "std": rec["metrics"][metric].get("std", np.nan),
                "min": rec["metrics"][metric].get("min", np.nan),
                "max": rec["metrics"][metric].get("max", np.nan),
                "median": rec["metrics"][metric].get("median", np.nan),
            }
            data[analysis][metric].append(entry)

    # Sort by latent dimension
    for analysis in analysis_names:
        for metric in metric_names:
            data[analysis][metric] = sorted(data[analysis][metric], key=lambda x: x["latent_dim"])

    all_latent_dims = sorted(set(rec["latent_dim"] for rec in parsed))

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    for ax, metric in zip(axes, metric_names):
        plotted_mean_values = []
        for analysis in analysis_names:
            entries = data[analysis][metric]
            if len(entries) == 0:
                continue

            x = np.array([e["latent_dim"] for e in entries], dtype=float)
            y = np.array([e["mean"] for e in entries], dtype=float)
            plotted_mean_values.append(y)
            ax.plot(x, y, marker="o", label=analysis)

            # if band_mode == "std":
            y_low = np.array([e["mean"] - e["std"] for e in entries], dtype=float)
            y_high = np.array([e["mean"] + e["std"] for e in entries], dtype=float)
            ax.fill_between(x, y_low, y_high, alpha=0.2)

            # elif band_mode == "minmax":
            #     y_low = np.array([e["min"] for e in entries], dtype=float)
            #     y_high = np.array([e["max"] for e in entries], dtype=float)
            #     ax.fill_between(x, y_low, y_high, alpha=0.2)
        
            if len(plotted_mean_values) > 0:
                all_means = np.concatenate(plotted_mean_values)
                y_min = np.min(all_means)*0.8
                y_max = np.max(all_means)*1.1
                if y_min == y_max:
                    margin = 0.05 * max(abs(y_min), 1.0)
                else:
                    margin = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes:
        if xscale == "log":
            ax.set_xscale("log")
        elif xscale == "linear":
            ax.set_xscale("linear")
        else:
            raise ValueError("xscale must be 'log' or 'linear'")
        ax.set_xticks(all_latent_dims)
        ax.set_xticklabels([str(d) for d in all_latent_dims])

    axes[-1].set_xlabel("Latent dimension / number of principal components")

    title_parts = ["AE3d vs PCA"]
    if n_patients is not None:
        title_parts.append(f"{n_patients} training patients")
    if splitname is not None:
        title_parts.append(f"split={splitname}")
    if ae_epochs is not None:
        title_parts.append(f"AE={ae_epochs} epochs")
    # if band_mode is not None:
    #     title_parts.append(f"band={band_mode}")

    fig.suptitle(" | ".join(title_parts), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if vline_dim is not None:
        for ax in axes:
            ax.axvline(x=vline_dim, color="red", linestyle="--", linewidth=1.5)

    save_path = path_resultsfolder + "/AE3d_vs_PCA_" + splitname + "_" + repr(n_patients) + "patients_metrics.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes, parsed

def plot_ae_loss_from_txt(loss_txt_path, save_path=None):
    """
    Read an AE loss txt file with lines like:
        Epoch 1: 0.02000822302652523

    Then:
    - extract epoch number and loss
    - plot loss vs epoch
    - x axis linear
    - y axis log
    - add exponential fit curve
    - save figure

    Parameters
    ----------
    loss_txt_path : str
        Path to the loss txt file.
    save_path : str or None
        Output path for the figure. If None, uses the input filename and
        replaces '.txt' by '_lossepochplot.png'.

    Returns
    -------
    epochs : np.ndarray
    losses : np.ndarray
    fit_coeffs : tuple
        (a, b) in log(loss) = a * epoch + b
    """
    epoch_list = []
    loss_list = []

    pattern = re.compile(r"Epoch\s+(\d+)\s*:\s*([0-9eE\.\+\-]+)")

    with open(loss_txt_path, "r") as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                epoch_list.append(int(match.group(1)))
                loss_list.append(float(match.group(2)))

    if len(epoch_list) == 0:
        raise ValueError(f"No valid 'Epoch k: loss' lines found in {loss_txt_path}")

    epochs = np.array(epoch_list, dtype=float)
    losses = np.array(loss_list, dtype=float)

    # Exponential fit:
    # loss(epoch) ~= exp(a * epoch + b)
    # so log(loss) = a * epoch + b
    positive_mask = losses > 0
    if np.sum(positive_mask) < 2:
        raise ValueError("Need at least two strictly positive loss values for exponential fit.")

    fit_epochs = epochs[positive_mask]
    fit_losses = losses[positive_mask]

    a, b = np.polyfit(fit_epochs, np.log(fit_losses), deg=1)
    fitted_losses = np.exp(a * epochs + b)

    # Save path
    if save_path is None:
        if loss_txt_path.endswith(".txt"):
            save_path = loss_txt_path[:-4] + "_lossepochplot.png"
        else:
            save_path = loss_txt_path + "_lossepochplot.png"

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, losses, marker="o", linewidth=1.5, label="Training loss")
    ax.plot(
        epochs,
        fitted_losses,
        linestyle="--",
        linewidth=2,
        label=f"Exponential fit (log-loss slope = {a:.4e})"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss")
    ax.set_title("loss as a function of training epoch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return epochs, losses, (a, b)