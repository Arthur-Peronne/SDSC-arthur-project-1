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
import time

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

class AutoEncoder3D_Current(nn.Module):
    """
    Original AE model:
    (1,32,128,128)
    -> (8,16,64,64)
    -> (16,8,32,32)
    -> (32,4,16,16)
    -> bottleneck conv -> (64,4,16,16)
    -> flatten 65536 -> latent_dim
    """

    def __init__(self, latent_dim=10, input_shape=(1, 32, 128, 128)):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.enc1 = Conv3DBlock(in_channels=1, out_channels=8, downsample=True)
        self.enc2 = Conv3DBlock(in_channels=8, out_channels=16, downsample=True)
        self.enc3 = Conv3DBlock(in_channels=16, out_channels=32, downsample=True)

        self.bottleneck_conv = Conv3DBlock(in_channels=32, out_channels=64, downsample=False)

        self.feature_shape = (64, 4, 16, 16)
        flattened_size = 64 * 4 * 16 * 16  # 65536

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.dec1 = UpConv3DBlock(in_channels=64, out_channels=32)
        self.dec2 = UpConv3DBlock(in_channels=32, out_channels=16)
        self.dec3 = UpConv3DBlock(in_channels=16, out_channels=8)

        self.final_conv = nn.Conv3d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

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
        x = x.view(-1, *self.feature_shape)

        x = self.dec1(x)              # -> (B, 32, 8, 32, 32)
        x = self.dec2(x)              # -> (B, 16, 16, 64, 64)
        x = self.dec3(x)              # -> (B, 8, 32, 128, 128)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_FCDeep(nn.Module):
    """
    Model A:
    Progressive compression down to (128,1,4,4),
    then flatten -> latent vector -> linear decode.
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128)):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.enc1 = Conv3DBlock(1, 8, downsample=True)      # -> (8,16,64,64)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)     # -> (16,8,32,32)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)    # -> (32,4,16,16)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)    # -> (64,2,8,8)

        # Last compression without isotropic pooling because depth=2
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )                                                   # -> (128,2,8,8)

        self.final_down = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )                                                   # -> (128,1,4,4)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.initial_up = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )                                                   # -> (128,2,8,8)

        self.dec1 = UpConv3DBlock(128, 64)   # -> (64,4,16,16)
        self.dec2 = UpConv3DBlock(64, 32)    # -> (32,8,32,32)
        self.dec3 = UpConv3DBlock(32, 16)    # -> (16,16,64,64)
        self.dec4 = UpConv3DBlock(16, 8)     # -> (8,32,128,128)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_Conv(nn.Module):
    """
    Model B:
    Fully convolutional bottleneck with shape (C,1,2,2),
    where latent_dim = 4 * C.
    No linear layers.
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128)):
        super().__init__()

        if latent_dim % 4 != 0:
            raise ValueError("For AutoEncoder3D_Conv, latent_dim must be a multiple of 4.")

        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.latent_channels = latent_dim // 4

        # Encoder
        self.enc1 = Conv3DBlock(1, 8, downsample=True)      # -> (8,16,64,64)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)     # -> (16,8,32,32)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)    # -> (32,4,16,16)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)    # -> (64,2,8,8)

        # Bottleneck reduction:
        # (64,2,8,8) -> (128,2,8,8) -> (128,1,4,4) -> (C,1,2,2)
        self.pre_latent = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.reduce_to_1x4x4 = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2)
        )   # (128,2,8,8) -> (128,1,4,4)

        self.reduce_to_latent = nn.Conv3d(
            in_channels=128,
            out_channels=self.latent_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )   # (128,1,4,4) -> (C,1,2,2)

        # Decoder bottleneck inverse:
        # (C,1,2,2) -> (128,1,4,4) -> (128,2,8,8)
        self.expand_from_latent = nn.Sequential(
            nn.ConvTranspose3d(
                self.latent_channels,
                128,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )   # -> (128,1,4,4)

        self.expand_to_2x8x8 = nn.Sequential(
            nn.ConvTranspose3d(
                128,
                128,
                kernel_size=(2, 2, 2),
                stride=(2, 2, 2)
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )   # -> (128,2,8,8)

        self.dec1 = UpConv3DBlock(128, 64)   # -> (64,4,16,16)
        self.dec2 = UpConv3DBlock(64, 32)    # -> (32,8,32,32)
        self.dec3 = UpConv3DBlock(32, 16)    # -> (16,16,64,64)
        self.dec4 = UpConv3DBlock(16, 8)     # -> (8,32,128,128)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode_tensor(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.pre_latent(x)           # -> (128,2,8,8)
        x = self.reduce_to_1x4x4(x)      # -> (128,1,4,4)
        z_tensor = self.reduce_to_latent(x)  # -> (B,C,1,2,2)

        return z_tensor

    def encode(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)  # -> (B, latent_dim)
        return z

    def decode_from_tensor(self, z_tensor):
        x = self.expand_from_latent(z_tensor)   # -> (128,1,4,4)
        x = self.expand_to_2x8x8(x)             # -> (128,2,8,8)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def decode(self, z):
        z_tensor = z.view(-1, self.latent_channels, 1, 2, 2)
        x = self.decode_from_tensor(z_tensor)
        return x

    def forward(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)
        x_recon = self.decode_from_tensor(z_tensor)
        return x_recon, z

def build_autoencoder(model_name, latent_dimensions):
    """
    Build one of the available AE models.
    """
    if model_name == "AE3dCurrent":
        return AutoEncoder3D_Current(latent_dim=latent_dimensions)

    elif model_name == "AE3dFCDeep":
        return AutoEncoder3D_FCDeep(latent_dim=latent_dimensions)

    elif model_name == "AE3dConv":
        return AutoEncoder3D_Conv(latent_dim=latent_dimensions)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def reconstruction_metrics(x_true, x_pred, patient_number, simulation_name, n_epochs, metrics_dataset, savemetrics=True):
    """
    Compute reconstruction metrics between two 3D images
    of identical shape.
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
                path_tempodata_folder
                + "autoencoder/"
                + simulation_name
                + f"_resultspatient{patient_number}_{metrics_dataset}.txt"
            )
        else:
            save_path = (
                path_tempodata_folder
                + "autoencoder/"
                + simulation_name
                + f"_{n_epochs}epochs_resultspatient{patient_number}_{metrics_dataset}.txt"
            )

        with open(save_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    return metrics

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
        path_tempodata_folder
        + "autoencoder/"
        + simulation_name
        + f"_{n_epochs}epochs.pth"
    )

    loss_path = (
        path_tempodata_folder
        + "autoencoder/"
        + simulation_name
        + f"_{n_epochs}epochs_loss.txt"
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
                    path_tempodata_folder
                    + "autoencoder/"
                    + simulation_name
                    + f"_{current_epoch}epochs.pth"
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
    X_ini = pef.get_vectorsarray(
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

# def ae_reconstructX(patient_tensor, X_maxnorm, model):
#     """
#     """
#     # Get patient initial X and its reconstruction from the model
#     x_patient = patient_tensor[0].unsqueeze(0) 
#     with torch.no_grad():
#         x_recon, z = model(x_patient)
#     # Put back the right dimensions and de-norm
#     x_patient_3d = x_patient.squeeze(0).squeeze(0).cpu().numpy()   # -> (32, 128, 128)
#     x_patient_3d = np.transpose(x_patient_3d, (1, 2, 0))           # -> (128, 128, 32)
#     x_ini_denorm = x_patient_3d * X_maxnorm 
#     x_recon_np = x_recon.squeeze(0).squeeze(0).cpu().numpy()   # -> (32, 128, 128)
#     x_recon_np = np.transpose(x_recon_np, (1, 2, 0))           # -> (128, 128, 32)
#     x_recon_denorm = x_recon_np * X_maxnorm 
#     return x_ini_denorm, x_recon_denorm

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
                path_tempodata_folder
                + "autoencoder/"
                + simulation_name
                + f"_summarymetrics_{metrics_dataset}.txt"
            )
        else:
            save_path = (
                path_tempodata_folder
                + "autoencoder/"
                + simulation_name
                + f"_{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt"
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


def ae_plotcompare_selected(
    patient_numbers,
    n_patients,
    train_dataset,
    test_dataset,
    X_maxnorm,
    model,
    splitname,
    latent_dimensions,
    n_epochs,
    metrics_dataset="test"
):
    """
    Plot reconstruction for selected patients, either from train or test dataset.
    """

    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train' or 'test'")

        for patient_number in patient_numbers:

            if metrics_dataset == "test":
                dataset = test_dataset
                idx = patient_number - (n_patients + 1)

            elif metrics_dataset == "validation":
                dataset = validation_dataset
                if dataset is None:
                    raise ValueError("validation_dataset is None but metrics_dataset='validation'")
                idx = patient_number - (n_patients + 1)

            else:  # train
                dataset = train_dataset
                idx = patient_number - 1
    
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Invalid index computed: {idx} for patient {patient_number}")

        patient_tensor = dataset[idx]

        x_patient, x_recon_denorm = ae_reconstructX(
            patient_tensor, X_maxnorm, model
        )

        ae_plotcompare_onepatient(
            x_recon_denorm,
            patient_number,
            "registered_framesBIS",
            splitname,
            latent_dimensions,
            details_rec=f"AErec_{n_epochs}epochs_{metrics_dataset}"
        )

def _parse_summarymetrics_file(filepath):
    """
    Parse one summary metrics txt file.

    Supported filename patterns:
    - AE3dCurrent_<N>patients_<splitname>_<D>dims_<E>epochs_summarymetrics_<train/validation/test>.txt
    - AE3dFCDeep_<N>patients_<splitname>_<D>dims_<E>epochs_summarymetrics_<train/validation/test>.txt
    - AE3dConv_<N>patients_<splitname>_<D>dims_<E>epochs_summarymetrics_<train/validation/test>.txt
    - PCA_<N>patients_<splitname>_<D>dims_summarymetrics_<train/validation/test>.txt
    """
    filename = os.path.basename(filepath)

    pattern_ae = (
        r'^(AE3dCurrent|AE3dFCDeep|AE3dConv)_'      # analysis
        r'(\d+)patients_'                           # n_patients
        r'(.+?)_'                                   # splitname
        r'(\d+)dims_'                               # latent_dim
        r'(\d+)epochs_'                             # n_epochs
        r'summarymetrics_(train|validation|test)\.txt$'
    )

    pattern_pca = (
        r'^(PCA)_'
        r'(\d+)patients_'
        r'(.+?)_'
        r'(\d+)dims_'
        r'summarymetrics_(train|validation|test)\.txt$'
    )

    match_ae = re.match(pattern_ae, filename)
    match_pca = re.match(pattern_pca, filename)

    if match_ae is not None:
        analysis = match_ae.group(1)
        n_patients = int(match_ae.group(2))
        splitname = match_ae.group(3)
        latent_dim = int(match_ae.group(4))
        n_epochs = int(match_ae.group(5))
        metrics_dataset = match_ae.group(6)

        device_tag = None
        batch_size = None

    elif match_pca is not None:
        analysis = match_pca.group(1)
        n_patients = int(match_pca.group(2))
        splitname = match_pca.group(3)
        latent_dim = int(match_pca.group(4))
        metrics_dataset = match_pca.group(5)

        device_tag = None
        batch_size = None
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

            if line in {"MSE", "RMSE", "MAE", "R2"}:
                current_metric = line
                metrics[current_metric] = {}
                continue

            if current_metric is not None and ":" in line:
                key, value = line.split(":", 1)
                metrics[current_metric][key.strip()] = float(value.strip())

    return {
        "analysis": analysis,
        "n_patients": n_patients,
        "splitname": splitname,
        "latent_dim": latent_dim,
        "device_tag": device_tag,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "metrics_dataset": metrics_dataset,
        "metrics": metrics,
        "filepath": filepath,
    }

def plot_summarymetrics_vs_latentdim(
    results_folder,
    splitname=None,
    n_patients=None,
    ae_epochs_list=None,
    xscale="log",
    vline_dim=None,
    device_tag=None,
    batch_size=None,
    metrics_dataset=None,
    band_mode=None,
):
    """
    Plot MSE, RMSE, MAE, and R2 vs latent dimension, with:
    - 1 PCA curve
    - 1 AE3d curve for each AE epoch count found (or requested)

    Parameters
    ----------
    results_folder : str
        Folder containing files like:
        - AE3d_120patients_split0_119dims_GPU_4batch_300epochs_summarymetrics_test.txt
        - PCA_120patients_split0_119dims_summarymetrics_test.txt
    splitname : str or None
        If provided, only files matching this splitname are used.
    n_patients : int or None
        If provided, only files matching this training patient count are used.
    ae_epochs_list : list[int] or None
        AE epoch counts to include.
        If None, include all AE epoch counts found in the folder.
    xscale : {"log", "linear"}
        Scale for x axis.
    vline_dim : int or None
        Optional vertical reference line.
    device_tag : str or None
        If provided, only AE3d files matching this device tag are used.
    batch_size : int or None
        If provided, only AE3d files matching this batch size are used.
    metrics_dataset : {"train", "test"} or None
        If provided, only files matching this metrics dataset are used.
    band_mode : {"std", "minmax", None}
        Optional uncertainty band around the mean curve.

    Returns
    -------
    fig, axes, parsed_results
    """
    if band_mode not in {"std", "minmax", None}:
        raise ValueError("band_mode must be one of: 'std', 'minmax', None")

    filepaths = sorted(glob.glob(os.path.join(results_folder, "*_summarymetrics_*.txt")))

    if len(filepaths) == 0:
        raise FileNotFoundError(f"No *_summarymetrics_*.txt files found in: {results_folder}")

    parsed = []
    for fp in filepaths:
        try:
            rec = _parse_summarymetrics_file(fp)
        except ValueError:
            continue

        if splitname is not None and rec["splitname"] != splitname:
            continue
        if n_patients is not None and rec["n_patients"] != n_patients:
            continue
        if metrics_dataset is not None and rec["metrics_dataset"] != metrics_dataset:
            continue

        # AE-only filters
        if rec["analysis"] == "AE3d":
            if ae_epochs_list is not None and rec["n_epochs"] not in ae_epochs_list:
                continue
            if device_tag is not None and rec["device_tag"] != device_tag:
                continue
            if batch_size is not None and rec["batch_size"] != batch_size:
                continue

        parsed.append(rec)

    if len(parsed) == 0:
        raise ValueError("No matching summary metrics files found after filtering.")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]

    # Find all AE epoch counts present after filtering
    ae_epochs_found = sorted(
        set(rec["n_epochs"] for rec in parsed if rec["analysis"] == "AE3d")
    )

    # Build series names
    series_names = ["PCA"] + [f"AE3d_{ep}epochs" for ep in ae_epochs_found]

    # Organize data
    data = {series: {metric: [] for metric in metric_names} for series in series_names}

    for rec in parsed:
        latent_dim = rec["latent_dim"]

        if rec["analysis"] == "PCA":
            series_name = "PCA"
        elif rec["analysis"] == "AE3d":
            series_name = f"AE3d_{rec['n_epochs']}epochs"
        else:
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
            data[series_name][metric].append(entry)

    # Sort by latent dimension
    for series in series_names:
        for metric in metric_names:
            data[series][metric] = sorted(data[series][metric], key=lambda x: x["latent_dim"])

    all_latent_dims = sorted(set(rec["latent_dim"] for rec in parsed))

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    for ax, metric in zip(axes, metric_names):
        plotted_mean_values = []

        for series in series_names:
            entries = data[series][metric]
            if len(entries) == 0:
                continue

            x = np.array([e["latent_dim"] for e in entries], dtype=float)
            y = np.array([e["mean"] for e in entries], dtype=float)
            plotted_mean_values.append(y)

            ax.plot(x, y, marker="o", label=series)

            if band_mode == "std":
                y_low = np.array([e["mean"] - e["std"] for e in entries], dtype=float)
                y_high = np.array([e["mean"] + e["std"] for e in entries], dtype=float)
                ax.fill_between(x, y_low, y_high, alpha=0.15)

            elif band_mode == "minmax":
                y_low = np.array([e["min"] for e in entries], dtype=float)
                y_high = np.array([e["max"] for e in entries], dtype=float)
                ax.fill_between(x, y_low, y_high, alpha=0.15)

        if len(plotted_mean_values) > 0:
            all_means = np.concatenate(plotted_mean_values)
            y_min = np.min(all_means) * 0.8
            y_max = np.max(all_means) * 1.1
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
    if device_tag is not None:
        title_parts.append(f"device={device_tag}")
    if batch_size is not None:
        title_parts.append(f"batch={batch_size}")
    if metrics_dataset is not None:
        title_parts.append(f"metrics={metrics_dataset}")
    if ae_epochs_list is not None:
        title_parts.append("AE epochs=" + ",".join(str(v) for v in sorted(ae_epochs_list)))
    if band_mode is not None:
        title_parts.append(f"band={band_mode}")

    fig.suptitle(" | ".join(title_parts), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if vline_dim is not None:
        for ax in axes:
            ax.axvline(x=vline_dim, color="red", linestyle="--", linewidth=1.5)

    dataset_str = metrics_dataset if metrics_dataset is not None else "all"
    save_path = (
        path_resultsfolder
        + "/AE3d_vs_PCA_metrics_"
        + dataset_str
        + "_"
        + (splitname if splitname is not None else "allsplits")
        + "_"
        + (repr(n_patients) + "patients" if n_patients is not None else "allpatients")
        + ".png"
    )

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

def ae_build_basename(
    model_name,
    n_patients,
    splitname,
    latent_dim,
    n_epochs=None
):
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

def dataset_for_metrics(metrics_dataset, train_dataset, validation_dataset, test_dataset, n_train, n_validation=0):
    if metrics_dataset == "train":
        dataset_for_metrics = train_dataset
        patient_offset = 0

    elif metrics_dataset == "validation":
        if validation_dataset is None:
            raise ValueError("validation_dataset is None but metrics_dataset='validation'")
        dataset_for_metrics = validation_dataset
        patient_offset = n_train

    elif metrics_dataset == "test":
        dataset_for_metrics = test_dataset
        patient_offset = n_train + n_validation

    else:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")

    return dataset_for_metrics, patient_offset

def plot_r2_test_vs_train(
    results_folder,
    splitname=None,
    n_patients=None,
    ae_epochs_list=None,
    device_tag=None,
    batch_size=None,
    annotate_dims=True,
    dims_to_annotate=None,
    save_path=None,
):
    """
    Plot test R2 versus train R2.

    - One PCA curve
    - One AE3d curve for each AE epoch count found (or requested)
    - Each latent dimension is one point
    - Optional point labels with the latent dimension

    Parameters
    ----------
    results_folder : str
        Folder containing summary metric files like:
        - AE3d_120patients_split0_119dims_GPU_4batch_300epochs_summarymetrics_train.txt
        - AE3d_120patients_split0_119dims_GPU_4batch_300epochs_summarymetrics_test.txt
        - PCA_120patients_split0_119dims_summarymetrics_train.txt
        - PCA_120patients_split0_119dims_summarymetrics_test.txt
    splitname : str or None
        Filter by splitname.
    n_patients : int or None
        Filter by training patient count.
    ae_epochs_list : list[int] or None
        AE epoch counts to include. If None, include all AE epoch counts found.
    device_tag : str or None
        AE-only filter.
    batch_size : int or None
        AE-only filter.
    annotate_dims : bool
        Whether to label points with latent dimensions.
    dims_to_annotate : list[int] or None
        If provided, only annotate these dimensions.
    save_path : str or None
        Optional explicit output path.

    Returns
    -------
    fig, ax, paired_records
    """
    filepaths = sorted(glob.glob(os.path.join(results_folder, "*_summarymetrics_*.txt")))

    if len(filepaths) == 0:
        raise FileNotFoundError(f"No *_summarymetrics_*.txt files found in: {results_folder}")

    parsed = []
    for fp in filepaths:
        try:
            rec = _parse_summarymetrics_file(fp)
        except ValueError:
            continue

        if splitname is not None and rec["splitname"] != splitname:
            continue
        if n_patients is not None and rec["n_patients"] != n_patients:
            continue

        if rec["analysis"] == "AE3d":
            if ae_epochs_list is not None and rec["n_epochs"] not in ae_epochs_list:
                continue
            if device_tag is not None and rec["device_tag"] != device_tag:
                continue
            if batch_size is not None and rec["batch_size"] != batch_size:
                continue

        parsed.append(rec)

    if len(parsed) == 0:
        raise ValueError("No matching summary metrics files found after filtering.")

    # Pair train/test files by common key
    paired_records = {}

    for rec in parsed:
        if rec["analysis"] == "AE3d":
            key = (
                rec["analysis"],
                rec["n_patients"],
                rec["splitname"],
                rec["latent_dim"],
                rec["device_tag"],
                rec["batch_size"],
                rec["n_epochs"],
            )
        elif rec["analysis"] == "PCA":
            key = (
                rec["analysis"],
                rec["n_patients"],
                rec["splitname"],
                rec["latent_dim"],
            )
        else:
            continue

        if key not in paired_records:
            paired_records[key] = {}

        paired_records[key][rec["metrics_dataset"]] = rec

    # Build series
    ae_epochs_found = sorted(
        set(
            rec["n_epochs"]
            for rec in parsed
            if rec["analysis"] == "AE3d"
        )
    )

    series_names = ["PCA"] + [f"AE3d_{ep}epochs" for ep in ae_epochs_found]
    series_data = {series: [] for series in series_names}

    for key, pair in paired_records.items():
        if "train" not in pair or "test" not in pair:
            continue

        train_rec = pair["train"]
        test_rec = pair["test"]

        if "R2" not in train_rec["metrics"] or "R2" not in test_rec["metrics"]:
            continue

        x_train = train_rec["metrics"]["R2"].get("mean", np.nan)
        y_test = test_rec["metrics"]["R2"].get("mean", np.nan)
        latent_dim = train_rec["latent_dim"]

        if train_rec["analysis"] == "PCA":
            series_name = "PCA"
        else:
            series_name = f"AE3d_{train_rec['n_epochs']}epochs"

        series_data[series_name].append({
            "latent_dim": latent_dim,
            "r2_train": x_train,
            "r2_test": y_test,
        })

    # Sort each series by latent dimension
    for series in series_names:
        series_data[series] = sorted(series_data[series], key=lambda d: d["latent_dim"])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    for series in series_names:
        entries = series_data[series]
        if len(entries) == 0:
            continue

        x = np.array([e["r2_train"] for e in entries], dtype=float)
        y = np.array([e["r2_test"] for e in entries], dtype=float)

        ax.plot(x, y, marker="o", label=series)

        if annotate_dims:
            for e in entries:
                dim = e["latent_dim"]
                if dims_to_annotate is None or dim in dims_to_annotate:
                    ax.annotate(
                        str(dim),
                        (e["r2_train"], e["r2_test"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8
                    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # Collect all plotted values
    all_x = []
    all_y = []

    for series in series_names:
        entries = series_data[series]
        if len(entries) == 0:
            continue

        all_x.extend([e["r2_train"] for e in entries])
        all_y.extend([e["r2_test"] for e in entries])

    if len(all_x) > 0 and len(all_y) > 0:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # Add margins
        x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.05
        y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.05

        ax.set_xlim(max(0, x_min - x_margin), 1)
        ax.set_ylim(max(0, y_min - y_margin), 1)

    ax.set_xlabel("Train R2")
    ax.set_ylabel("Test R2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    title_parts = ["Test R2 vs Train R2"]
    if n_patients is not None:
        title_parts.append(f"{n_patients} training patients")
    if splitname is not None:
        title_parts.append(f"split={splitname}")
    if device_tag is not None:
        title_parts.append(f"device={device_tag}")
    if batch_size is not None:
        title_parts.append(f"batch={batch_size}")
    if ae_epochs_list is not None:
        title_parts.append("AE epochs=" + ",".join(str(v) for v in sorted(ae_epochs_list)))

    ax.set_title(" | ".join(title_parts))

    fig.tight_layout()

    if save_path is None:
        save_path = (
            path_resultsfolder
            + "/R2_test_vs_train_"
            + (splitname if splitname is not None else "allsplits")
            + "_"
            + (repr(n_patients) + "patients" if n_patients is not None else "allpatients")
            + ".png"
        )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax, paired_records

def plot_ae_metrics_vs_latentdim_by_architecture(
    results_folder,
    splitname=None,
    n_patients=None,
    metrics_dataset="validation",
    model_names=None,
    epoch_list=None,
    xscale="linear",
    band_mode=None,
    save_prefix="AEcompare_byArchitecture"
):
    """
    For each AE architecture, plot metrics vs latent dimension.

    One figure per architecture.
    In each figure:
    - x-axis = latent dimension
    - one curve per epoch
    - panels = MSE, RMSE, MAE, R2
    - optional confidence band from std or min/max

    Parameters
    ----------
    results_folder : str
        Folder containing AE summary metric files.
    splitname : str or None
        Filter by splitname.
    n_patients : int or None
        Filter by training patient count.
    metrics_dataset : {"train", "validation", "test"}
        Which dataset summary files to use.
    model_names : list[str] or None
        AE model names to include. If None, use all found.
    epoch_list : list[int] or None
        Epochs to include. If None, use all found.
    xscale : {"linear", "log"}
        X-axis scale.
    band_mode : {None, "std", "minmax"}
        Optional uncertainty band around the mean curve.
    save_prefix : str
        Prefix for saved figure names.

    Returns
    -------
    figures : dict
        {model_name: (fig, axes, parsed_subset)}
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")
    if band_mode not in {None, "std", "minmax"}:
        raise ValueError("band_mode must be one of: None, 'std', 'minmax'")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]

    filepaths = sorted(glob.glob(os.path.join(results_folder, "*_summarymetrics_*.txt")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No *_summarymetrics_*.txt files found in: {results_folder}")

    parsed = []
    for fp in filepaths:
        try:
            rec = _parse_summarymetrics_file(fp)
        except ValueError:
            continue

        if rec["analysis"] not in {"AE3dCurrent", "AE3dFCDeep", "AE3dConv"}:
            continue
        if splitname is not None and rec["splitname"] != splitname:
            continue
        if n_patients is not None and rec["n_patients"] != n_patients:
            continue
        if rec["metrics_dataset"] != metrics_dataset:
            continue
        if model_names is not None and rec["analysis"] not in model_names:
            continue
        if epoch_list is not None and rec["n_epochs"] not in epoch_list:
            continue

        parsed.append(rec)

    if len(parsed) == 0:
        raise ValueError("No matching AE summary metric files found after filtering.")

    models_found = sorted(set(rec["analysis"] for rec in parsed))
    figures = {}

    for model_name in models_found:
        subset = [rec for rec in parsed if rec["analysis"] == model_name]
        epochs_found = sorted(set(rec["n_epochs"] for rec in subset))

        data = {
            ep: {metric: [] for metric in metric_names}
            for ep in epochs_found
        }

        for rec in subset:
            latent_dim = rec["latent_dim"]
            ep = rec["n_epochs"]

            for metric in metric_names:
                if metric not in rec["metrics"]:
                    continue

                data[ep][metric].append({
                    "latent_dim": latent_dim,
                    "mean": rec["metrics"][metric].get("mean", np.nan),
                    "std": rec["metrics"][metric].get("std", np.nan),
                    "min": rec["metrics"][metric].get("min", np.nan),
                    "max": rec["metrics"][metric].get("max", np.nan),
                    "median": rec["metrics"][metric].get("median", np.nan),
                })

        for ep in epochs_found:
            for metric in metric_names:
                data[ep][metric] = sorted(data[ep][metric], key=lambda d: d["latent_dim"])

        all_latent_dims = sorted(set(rec["latent_dim"] for rec in subset))

        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        for ax, metric in zip(axes, metric_names):
            plotted_y = []

            for ep in epochs_found:
                entries = data[ep][metric]
                if len(entries) == 0:
                    continue

                x = np.array([e["latent_dim"] for e in entries], dtype=float)
                y = np.array([e["mean"] for e in entries], dtype=float)

                line = ax.plot(x, y, marker="o", label=f"{ep} epochs")[0]
                plotted_y.append(y)

                if band_mode == "std":
                    y_low = np.array([e["mean"] - e["std"] for e in entries], dtype=float)
                    y_high = np.array([e["mean"] + e["std"] for e in entries], dtype=float)
                    ax.fill_between(x, y_low, y_high, alpha=0.15, color=line.get_color())

                elif band_mode == "minmax":
                    y_low = np.array([e["min"] for e in entries], dtype=float)
                    y_high = np.array([e["max"] for e in entries], dtype=float)
                    ax.fill_between(x, y_low, y_high, alpha=0.15, color=line.get_color())

            if len(plotted_y) > 0:
                all_y = np.concatenate(plotted_y)
                y_min = np.min(all_y)
                y_max = np.max(all_y)
                margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(abs(y_min), 1.0)
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
            ax.set_xticklabels([str(v) for v in all_latent_dims])

        axes[-1].set_xlabel("Latent dimension")

        title = (
            f"{model_name} | metrics={metrics_dataset}"
            + (f" | {n_patients} training patients" if n_patients is not None else "")
            + (f" | split={splitname}" if splitname is not None else "")
        )
        if band_mode is not None:
            title += f" | band={band_mode}"

        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = os.path.join(
            path_resultsfolder,
            f"{save_prefix}_{model_name}_{metrics_dataset}"
            + (f"_{band_mode}" if band_mode is not None else "")
            + (f"_{splitname}" if splitname is not None else "")
            + (f"_{n_patients}patients" if n_patients is not None else "")
            + ".png"
        )
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        figures[model_name] = (fig, axes, subset)

    return figures

def plot_ae_metrics_vs_latentdim_by_epoch(
    results_folder,
    splitname=None,
    n_patients=None,
    metrics_dataset="validation",
    model_names=None,
    epoch_list=None,
    xscale="linear",
    band_mode=None,
    save_prefix="AEcompare_byEpoch"
):
    """
    For each epoch, plot metrics vs latent dimension.

    One figure per epoch.
    In each figure:
    - x-axis = latent dimension
    - one curve per architecture
    - panels = MSE, RMSE, MAE, R2
    - optional confidence band from std or min/max

    Parameters
    ----------
    results_folder : str
        Folder containing AE summary metric files.
    splitname : str or None
        Filter by splitname.
    n_patients : int or None
        Filter by training patient count.
    metrics_dataset : {"train", "validation", "test"}
        Which dataset summary files to use.
    model_names : list[str] or None
        AE model names to include. If None, use all found.
    epoch_list : list[int] or None
        Epochs to include. If None, use all found.
    xscale : {"linear", "log"}
        X-axis scale.
    band_mode : {None, "std", "minmax"}
        Optional uncertainty band around the mean curve.
    save_prefix : str
        Prefix for saved figure names.

    Returns
    -------
    figures : dict
        {epoch: (fig, axes, parsed_subset)}
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")
    if band_mode not in {None, "std", "minmax"}:
        raise ValueError("band_mode must be one of: None, 'std', 'minmax'")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]

    filepaths = sorted(glob.glob(os.path.join(results_folder, "*_summarymetrics_*.txt")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No *_summarymetrics_*.txt files found in: {results_folder}")

    parsed = []
    for fp in filepaths:
        try:
            rec = _parse_summarymetrics_file(fp)
        except ValueError:
            continue

        if rec["analysis"] not in {"AE3dCurrent", "AE3dFCDeep", "AE3dConv"}:
            continue
        if splitname is not None and rec["splitname"] != splitname:
            continue
        if n_patients is not None and rec["n_patients"] != n_patients:
            continue
        if rec["metrics_dataset"] != metrics_dataset:
            continue
        if model_names is not None and rec["analysis"] not in model_names:
            continue
        if epoch_list is not None and rec["n_epochs"] not in epoch_list:
            continue

        parsed.append(rec)

    if len(parsed) == 0:
        raise ValueError("No matching AE summary metric files found after filtering.")

    epochs_found = sorted(set(rec["n_epochs"] for rec in parsed))
    figures = {}

    for ep in epochs_found:
        subset = [rec for rec in parsed if rec["n_epochs"] == ep]
        models_found = sorted(set(rec["analysis"] for rec in subset))

        data = {
            model_name: {metric: [] for metric in metric_names}
            for model_name in models_found
        }

        for rec in subset:
            latent_dim = rec["latent_dim"]
            model_name = rec["analysis"]

            for metric in metric_names:
                if metric not in rec["metrics"]:
                    continue

                data[model_name][metric].append({
                    "latent_dim": latent_dim,
                    "mean": rec["metrics"][metric].get("mean", np.nan),
                    "std": rec["metrics"][metric].get("std", np.nan),
                    "min": rec["metrics"][metric].get("min", np.nan),
                    "max": rec["metrics"][metric].get("max", np.nan),
                    "median": rec["metrics"][metric].get("median", np.nan),
                })

        for model_name in models_found:
            for metric in metric_names:
                data[model_name][metric] = sorted(data[model_name][metric], key=lambda d: d["latent_dim"])

        all_latent_dims = sorted(set(rec["latent_dim"] for rec in subset))

        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        for ax, metric in zip(axes, metric_names):
            plotted_y = []

            for model_name in models_found:
                entries = data[model_name][metric]
                if len(entries) == 0:
                    continue

                x = np.array([e["latent_dim"] for e in entries], dtype=float)
                y = np.array([e["mean"] for e in entries], dtype=float)

                line = ax.plot(x, y, marker="o", label=model_name)[0]
                plotted_y.append(y)

                if band_mode == "std":
                    y_low = np.array([e["mean"] - e["std"] for e in entries], dtype=float)
                    y_high = np.array([e["mean"] + e["std"] for e in entries], dtype=float)
                    ax.fill_between(x, y_low, y_high, alpha=0.15, color=line.get_color())

                elif band_mode == "minmax":
                    y_low = np.array([e["min"] for e in entries], dtype=float)
                    y_high = np.array([e["max"] for e in entries], dtype=float)
                    ax.fill_between(x, y_low, y_high, alpha=0.15, color=line.get_color())

            if len(plotted_y) > 0:
                all_y = np.concatenate(plotted_y)
                y_min = np.min(all_y)
                y_max = np.max(all_y)
                margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(abs(y_min), 1.0)
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
            ax.set_xticklabels([str(v) for v in all_latent_dims])

        axes[-1].set_xlabel("Latent dimension")

        title = (
            f"Epoch = {ep} | metrics={metrics_dataset}"
            + (f" | {n_patients} training patients" if n_patients is not None else "")
            + (f" | split={splitname}" if splitname is not None else "")
        )
        if band_mode is not None:
            title += f" | band={band_mode}"

        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = os.path.join(
            path_resultsfolder,
            f"{save_prefix}_{ep}epochs_{metrics_dataset}"
            + (f"_{band_mode}" if band_mode is not None else "")
            + (f"_{splitname}" if splitname is not None else "")
            + (f"_{n_patients}patients" if n_patients is not None else "")
            + ".png"
        )
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        figures[ep] = (fig, axes, subset)

    return figures

def plot_ae_r2_validation_vs_train_scatter(
    results_folder,
    splitname=None,
    n_patients=None,
    model_names=None,
    epoch_list=None,
    annotate_dims=True,
    save_prefix="AEcompare_R2_validation_vs_train"
):
    """
    Scatter plot of validation R2 vs train R2 for AE models only.

    Each point corresponds to one trained configuration:
    (architecture, latent_dim, n_epochs)

    Encoding:
    - color by architecture
    - marker by epoch
    - point label = latent_dim (optional)

    Parameters
    ----------
    results_folder : str
        Folder containing AE summary metric files.
    splitname : str or None
        Filter by splitname.
    n_patients : int or None
        Filter by training patient count.
    model_names : list[str] or None
        AE model names to include. If None, use all found.
    epoch_list : list[int] or None
        Epochs to include. If None, use all found.
    annotate_dims : bool
        Whether to annotate each point with its latent dimension.
    save_prefix : str
        Prefix for saved figure name.

    Returns
    -------
    fig, ax, paired_records
    """
    filepaths = sorted(glob.glob(os.path.join(results_folder, "*_summarymetrics_*.txt")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No *_summarymetrics_*.txt files found in: {results_folder}")

    parsed = []
    for fp in filepaths:
        try:
            rec = _parse_summarymetrics_file(fp)
        except ValueError:
            continue

        if rec["analysis"] not in {"AE3dCurrent", "AE3dFCDeep", "AE3dConv"}:
            continue
        if splitname is not None and rec["splitname"] != splitname:
            continue
        if n_patients is not None and rec["n_patients"] != n_patients:
            continue
        if model_names is not None and rec["analysis"] not in model_names:
            continue
        if epoch_list is not None and rec["n_epochs"] not in epoch_list:
            continue

        parsed.append(rec)

    if len(parsed) == 0:
        raise ValueError("No matching AE summary metric files found after filtering.")

    # Pair train/validation by common key
    paired_records = {}
    for rec in parsed:
        key = (
            rec["analysis"],
            rec["n_patients"],
            rec["splitname"],
            rec["latent_dim"],
            rec["n_epochs"],
        )

        if key not in paired_records:
            paired_records[key] = {}

        paired_records[key][rec["metrics_dataset"]] = rec

    valid_points = []
    for key, pair in paired_records.items():
        if "train" not in pair or "validation" not in pair:
            continue

        train_rec = pair["train"]
        val_rec = pair["validation"]

        if "R2" not in train_rec["metrics"] or "R2" not in val_rec["metrics"]:
            continue

        valid_points.append({
            "analysis": train_rec["analysis"],
            "n_patients": train_rec["n_patients"],
            "splitname": train_rec["splitname"],
            "latent_dim": train_rec["latent_dim"],
            "n_epochs": train_rec["n_epochs"],
            "r2_train": train_rec["metrics"]["R2"].get("mean", np.nan),
            "r2_validation": val_rec["metrics"]["R2"].get("mean", np.nan),
        })

    if len(valid_points) == 0:
        raise ValueError("No train/validation R2 pairs found after filtering.")

    models_found = sorted(set(p["analysis"] for p in valid_points))
    epochs_found = sorted(set(p["n_epochs"] for p in valid_points))

    # Fixed colors by architecture
    color_map = {
        "AE3dCurrent": "orange",
        "AE3dFCDeep": "green",
        "AE3dConv": "blue",
    }

    # Fixed markers by epoch
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_map = {}
    for i, ep in enumerate(epochs_found):
        marker_map[ep] = marker_cycle[i % len(marker_cycle)]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot points
    for p in valid_points:
        model_name = p["analysis"]
        ep = p["n_epochs"]

        ax.scatter(
            p["r2_train"],
            p["r2_validation"],
            color=color_map.get(model_name, "black"),
            marker=marker_map[ep],
            s=90
        )

        if annotate_dims:
            ax.annotate(
                str(p["latent_dim"]),
                (p["r2_train"], p["r2_validation"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8
            )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    all_x = np.array([p["r2_train"] for p in valid_points], dtype=float)
    all_y = np.array([p["r2_validation"] for p in valid_points], dtype=float)

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.05
    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.05

    ax.set_xlim(max(0, x_min - x_margin), min(1.0, x_max + x_margin))
    ax.set_ylim(max(0, y_min - y_margin), min(1.0, y_max + y_margin))

    ax.set_xlabel("Train R2")
    ax.set_ylabel("Validation R2")
    ax.grid(True, alpha=0.3)

    title = "Validation R2 vs Train R2 (AE only)"
    if n_patients is not None:
        title += f" | {n_patients} training patients"
    if splitname is not None:
        title += f" | split={splitname}"
    ax.set_title(title)

    # Build separate legends:
    # 1) architecture colors
    arch_handles = []
    for model_name in ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"]:
        if model_name in models_found:
            arch_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map[model_name],
                    markersize=10,
                    label=model_name
                )
            )

    # 2) epoch markers
    epoch_handles = []
    for ep in epochs_found:
        epoch_handles.append(
            plt.Line2D(
                [0], [0],
                marker=marker_map[ep],
                color="black",
                linestyle="None",
                markersize=10,
                label=f"{ep} epochs"
            )
        )

    legend1 = ax.legend(handles=arch_handles, title="Architecture", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=epoch_handles, title="Training length", loc="lower left")

    fig.tight_layout()

    save_path = os.path.join(
        path_resultsfolder,
        f"{save_prefix}"
        + (f"_{splitname}" if splitname is not None else "")
        + (f"_{n_patients}patients" if n_patients is not None else "")
        + ".png"
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax, paired_records