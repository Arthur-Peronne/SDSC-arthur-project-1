# src/visualization/ae_plots.py
"""
Plotting and result parsing functions for the 3D autoencoder.
"""

import re
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from src.config import RESULTS_FOLDER, TEMPODATA_FOLDER
from src.data import importdata as ipd
from src.visualization import mri_plots as mrp
from src.training.ae_training import ae_reconstructX


def ae_plotcompare_onepatient(x_recon_denorm, patient_number, folder_originals, splitname, latent_dimensions, details_rec="AErec"):
    """
    Plot original (ROI-masked) and reconstructed image for one patient.
    """
    ini_path = ipd.get_patient_modified_path(patient_number, folder_originals)
    ini_nii = nib.load(ini_path)
    reconstructed_nii = nib.Nifti1Image(x_recon_denorm.astype(np.float32), affine=ini_nii.affine, header=ini_nii.header.copy())
    inimask_path = ipd.get_patient_modified_path(patient_number, folder_originals, file_type="mask")
    inimask_nii = nib.load(inimask_path)
    ini_data = ini_nii.get_fdata(dtype=np.float32)
    mask_data = inimask_nii.get_fdata(dtype=np.float32)
    mask_bin = mask_data > 0
    ini_data_roi = ini_data.copy()
    ini_data_roi[~mask_bin] = 0.0
    ini_nii_roi = nib.Nifti1Image(ini_data_roi.astype(np.float32), affine=ini_nii.affine, header=ini_nii.header.copy())
    mrp.plot_oneimg(ini_nii_roi, cmapYN=False, patient_str="original", file_str="patient" + repr(patient_number), details_str="REGvoxROI")
    mrp.plot_oneimg(reconstructed_nii, cmapYN=False, cmap="", patient_str=splitname + "_" + repr(latent_dimensions) + "dims", file_str="patient" + repr(patient_number), details_str=details_rec)

def _decode_patient_number(patient_number, n_train_images, n_val_images, n_development, use_both_frames):
    """
    Convert internal patient_number back to real patient id and frame type.
    """
    if not use_both_frames:
        return patient_number, "ED"
    
    # Taille de chaque split en nombre de patients réels
    n_train_real = n_train_images // 2
    n_val_real   = n_val_images // 2
    
    # Index dans le dataset global (0-based)
    idx_global = patient_number - 1
    
    if idx_global < n_train_images:
        # Dans le train set
        split_size = n_train_real
        split_offset = 0
    elif idx_global < n_train_images + n_val_images:
        # Dans le val set
        split_size = n_val_real
        split_offset = n_train_real
        idx_global -= n_train_images
    else:
        # Dans le test set
        split_size = (150 - n_development)
        split_offset = n_development
        idx_global -= (n_train_images + n_val_images)
    
    if idx_global < split_size:
        # Frame ED
        real_patient = split_offset + idx_global + 1
        frame_type = "ED"
    else:
        # Frame ES
        real_patient = split_offset + (idx_global - split_size) + 1
        frame_type = "ES"
    
    return real_patient, frame_type

def ae_plotcompare_selected(
    patient_numbers,
    use_both_frames, 
    n_development, 
    n_train_images,
    n_val_images,
    train_dataset,
    test_dataset,
    X_maxnorm,
    model,
    model_name,
    splitname,
    latent_dimensions,
    n_epochs,
    metrics_dataset="test",
    validation_dataset=None
):
    """
    Plot reconstruction for selected patients, either from train or test dataset.
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train' or 'test'")

    for patient_number in patient_numbers:

        if metrics_dataset == "test":
            dataset = test_dataset
            idx = patient_number - (n_train_images + n_val_images + 1)
        elif metrics_dataset == "validation":
            dataset = validation_dataset
            if dataset is None:
                raise ValueError("validation_dataset is None but metrics_dataset='validation'")
            idx = patient_number - (n_train_images + 1)
        else:
            dataset = train_dataset
            idx = patient_number - 1

        real_patient, frame_type = _decode_patient_number(
            patient_number, n_train_images, n_val_images, n_development, use_both_frames
        )

        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Invalid index computed: {idx} for patient {patient_number}")

        patient_tensor = dataset[idx]
        x_patient, x_recon_denorm = ae_reconstructX(patient_tensor, X_maxnorm, model)

        ae_plotcompare_onepatient(
            x_recon_denorm,
            real_patient,
            "registered_frames",
            splitname,
            latent_dimensions,
            details_rec=f"{model_name}_{n_epochs}epochs_{metrics_dataset}_{frame_type}"
        )

def _parse_summarymetrics_file(filepath):
    """
    Parse one summary metrics txt file.
    """
    filename = Path(filepath).name

    pattern_ae = (
        r'^(AE3dCurrent|AE3dFCDeep|AE3dConv)_'
        r'(\d+)patients_'
        r'(.+?)_'
        r'(\d+)dims_'
        r'(\d+)epochs_'
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


def plot_ae_loss_from_txt(loss_txt_path, save_path=None):
    """
    Read an AE loss txt file and plot loss vs epoch with exponential fit.
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

    positive_mask = losses > 0
    if np.sum(positive_mask) < 2:
        raise ValueError("Need at least two strictly positive loss values for exponential fit.")

    fit_epochs = epochs[positive_mask]
    fit_losses = losses[positive_mask]

    a, b = np.polyfit(fit_epochs, np.log(fit_losses), deg=1)
    fitted_losses = np.exp(a * epochs + b)

    if save_path is None:
        loss_txt_path = str(loss_txt_path)
        if loss_txt_path.endswith(".txt"):
            save_path = loss_txt_path[:-4] + "_lossepochplot.png"
        else:
            save_path = loss_txt_path + "_lossepochplot.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", linewidth=1.5, label="Training loss")
    ax.plot(epochs, fitted_losses, linestyle="--", linewidth=2,
            label=f"Exponential fit (log-loss slope = {a:.4e})")
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
    Plot MSE, RMSE, MAE, and R2 vs latent dimension for PCA and AE models.
    """
    if band_mode not in {"std", "minmax", None}:
        raise ValueError("band_mode must be one of: 'std', 'minmax', None")

    # filepaths = sorted(glob.glob(str(Path(results_folder) / "*_summarymetrics_*.txt")))
    filepaths = sorted(glob.glob(str(Path(results_folder) / "**" / "*_summarymetrics_*.txt"), recursive=True))
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

    ae_epochs_found = sorted(set(rec["n_epochs"] for rec in parsed if rec["analysis"] == "AE3d"))
    series_names = ["PCA"] + [f"AE3d_{ep}epochs" for ep in ae_epochs_found]
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
    save_path = RESULTS_FOLDER / (
        f"AE3d_vs_PCA_metrics_{dataset_str}"
        f"_{splitname if splitname is not None else 'allsplits'}"
        f"_{str(n_patients) + 'patients' if n_patients is not None else 'allpatients'}.png"
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes, parsed


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
    Plot test R2 versus train R2 for PCA and AE models.
    """
    filepaths = sorted(glob.glob(str(Path(results_folder) / "**" / "*_summarymetrics_*.txt"), recursive=True))

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

    paired_records = {}
    for rec in parsed:
        if rec["analysis"] == "AE3d":
            key = (rec["analysis"], rec["n_patients"], rec["splitname"], rec["latent_dim"], rec["device_tag"], rec["batch_size"], rec["n_epochs"])
        elif rec["analysis"] == "PCA":
            key = (rec["analysis"], rec["n_patients"], rec["splitname"], rec["latent_dim"])
        else:
            continue

        if key not in paired_records:
            paired_records[key] = {}
        paired_records[key][rec["metrics_dataset"]] = rec

    ae_epochs_found = sorted(set(rec["n_epochs"] for rec in parsed if rec["analysis"] == "AE3d"))
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

    for series in series_names:
        series_data[series] = sorted(series_data[series], key=lambda d: d["latent_dim"])

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
                    ax.annotate(str(dim), (e["r2_train"], e["r2_test"]),
                                textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    all_x, all_y = [], []
    for series in series_names:
        entries = series_data[series]
        all_x.extend([e["r2_train"] for e in entries])
        all_y.extend([e["r2_test"] for e in entries])

    if len(all_x) > 0 and len(all_y) > 0:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
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
        save_path = RESULTS_FOLDER / (
            f"R2_test_vs_train"
            f"_{splitname if splitname is not None else 'allsplits'}"
            f"_{str(n_patients) + 'patients' if n_patients is not None else 'allpatients'}.png"
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
    One figure per architecture, one curve per epoch.
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")
    if band_mode not in {None, "std", "minmax"}:
        raise ValueError("band_mode must be one of: None, 'std', 'minmax'")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]

    filepaths = sorted(glob.glob(str(Path(results_folder) / "**" / "*_summarymetrics_*.txt"), recursive=True))
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

        data = {ep: {metric: [] for metric in metric_names} for ep in epochs_found}

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

        save_path = RESULTS_FOLDER / (
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
    One figure per epoch, one curve per architecture.
    """
    if metrics_dataset not in {"train", "validation", "test"}:
        raise ValueError("metrics_dataset must be 'train', 'validation', or 'test'")
    if band_mode not in {None, "std", "minmax"}:
        raise ValueError("band_mode must be one of: None, 'std', 'minmax'")

    metric_names = ["MSE", "RMSE", "MAE", "R2"]

    filepaths = sorted(glob.glob(str(Path(results_folder) / "**" / "*_summarymetrics_*.txt"), recursive=True))
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
        models_in_epoch = sorted(set(rec["analysis"] for rec in subset))

        data = {model: {metric: [] for metric in metric_names} for model in models_in_epoch}

        for rec in subset:
            latent_dim = rec["latent_dim"]
            model = rec["analysis"]

            for metric in metric_names:
                if metric not in rec["metrics"]:
                    continue
                data[model][metric].append({
                    "latent_dim": latent_dim,
                    "mean": rec["metrics"][metric].get("mean", np.nan),
                    "std": rec["metrics"][metric].get("std", np.nan),
                    "min": rec["metrics"][metric].get("min", np.nan),
                    "max": rec["metrics"][metric].get("max", np.nan),
                    "median": rec["metrics"][metric].get("median", np.nan),
                })

        for model in models_in_epoch:
            for metric in metric_names:
                data[model][metric] = sorted(data[model][metric], key=lambda d: d["latent_dim"])

        all_latent_dims = sorted(set(rec["latent_dim"] for rec in subset))

        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        for ax, metric in zip(axes, metric_names):
            plotted_y = []

            for model in models_in_epoch:
                entries = data[model][metric]
                if len(entries) == 0:
                    continue

                x = np.array([e["latent_dim"] for e in entries], dtype=float)
                y = np.array([e["mean"] for e in entries], dtype=float)
                line = ax.plot(x, y, marker="o", label=model)[0]
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
            f"{ep} epochs | metrics={metrics_dataset}"
            + (f" | {n_patients} training patients" if n_patients is not None else "")
            + (f" | split={splitname}" if splitname is not None else "")
        )
        if band_mode is not None:
            title += f" | band={band_mode}"

        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = RESULTS_FOLDER / (
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
    """
    filepaths = sorted(glob.glob(str(Path(results_folder) / "**" / "*_summarymetrics_*.txt"), recursive=True))
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

    paired_records = {}
    for rec in parsed:
        key = (rec["analysis"], rec["n_patients"], rec["splitname"], rec["latent_dim"], rec["n_epochs"])
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

    color_map = {
        "AE3dCurrent": "orange",
        "AE3dFCDeep": "green",
        "AE3dConv": "blue",
    }

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_map = {ep: marker_cycle[i % len(marker_cycle)] for i, ep in enumerate(epochs_found)}

    fig, ax = plt.subplots(figsize=(9, 9))

    for p in valid_points:
        ax.scatter(
            p["r2_train"], p["r2_validation"],
            color=color_map.get(p["analysis"], "black"),
            marker=marker_map[p["n_epochs"]],
            s=90
        )
        if annotate_dims:
            ax.annotate(str(p["latent_dim"]), (p["r2_train"], p["r2_validation"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)

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

    arch_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[m], markersize=10, label=m)
        for m in ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"] if m in models_found
    ]
    epoch_handles = [
        plt.Line2D([0], [0], marker=marker_map[ep], color="black",
                   linestyle="None", markersize=10, label=f"{ep} epochs")
        for ep in epochs_found
    ]

    legend1 = ax.legend(handles=arch_handles, title="Architecture", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=epoch_handles, title="Training length", loc="lower left")

    fig.tight_layout()

    save_path = RESULTS_FOLDER / (
        f"{save_prefix}"
        + (f"_{splitname}" if splitname is not None else "")
        + (f"_{n_patients}patients" if n_patients is not None else "")
        + ".png"
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax, paired_records

def plot_train_val_loss(
    loss_history,
    best_epoch,
    simulation_name,
    save_path=None,
    experiment_name="baseline",  
):
    """
    Plot training and validation loss curves from ae_training_early_stopping.
 
    Parameters
    ----------
    loss_history : dict
        {"train": [float, ...], "validation": [float, ...]}
        As returned by ae_training_early_stopping.
    best_epoch : int
        Epoch at which the best validation loss was reached. A vertical line
        is drawn at this epoch.
    simulation_name : str
        Used for the plot title and default save filename.
    save_path : Path or str or None
        If None, saves as {RESULTS_FOLDER}/{simulation_name}_train_val_loss.png
    """
    train_losses = np.array(loss_history["train"], dtype=float)
    val_losses = np.array(loss_history["validation"], dtype=float)
    epochs = np.arange(1, len(train_losses) + 1)
 
    if save_path is None:
        save_path = RESULTS_FOLDER / f"{simulation_name}_train_val_loss.png"
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(epochs, train_losses, linewidth=1.5, label="Train loss")
    ax.plot(epochs, val_losses, linewidth=1.5, label="Validation loss")
 
    ax.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Best epoch ({best_epoch})"
    )
 
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title(f"Train / validation loss — {simulation_name}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
 
    fig.tight_layout()

    # Save alongside model files
    model_save_path = (
        TEMPODATA_FOLDER / "autoencoder" / simulation_name / experiment_name / "train_val_loss.png"
    )
    fig.savefig(model_save_path, dpi=200, bbox_inches="tight")
    print(f"Loss plot saved: {model_save_path}")

    # # Save in results folder with full name
    # results_save_path = RESULTS_FOLDER / f"{simulation_name}_{experiment_name}_train_val_loss.png"
    # fig.savefig(results_save_path, dpi=200, bbox_inches="tight")
    # print(f"Loss plot saved: {results_save_path}")

    plt.close(fig)
 
    # print(f"Loss plot saved: {save_path}")

# New AE VS PCA plots for comparison

def _load_summarymetrics(folder, simulation_name, experiment_name, n_epochs, metrics_dataset):
    """
    Load a summarymetrics txt file and return a dict of metric -> mean value.
 
    Expected path (AE):
        folder / simulation_name / experiment_name / _{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt
 
    Expected path (PCA):
        folder / simulation_name / {experiment_name}_summarymetrics_{metrics_dataset}.txt
        (n_epochs is None for PCA)
 
    Returns
    -------
    dict : {metric_name: mean_value} or None if file not found
    """
    folder = Path(folder)
 
    if n_epochs is not None:
        path = folder / simulation_name / experiment_name / f"_{n_epochs}epochs_summarymetrics_{metrics_dataset}.txt"
    else:
        path = folder / simulation_name / f"{experiment_name}_summarymetrics_{metrics_dataset}.txt"
 
    if not path.exists():
        return None
 
    metrics = {}
    current_metric = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in {"MSE", "RMSE", "MAE", "R2"}:
                current_metric = line
                metrics[current_metric] = {}
            elif current_metric is not None and ":" in line:
                key, value = line.split(":", 1)
                metrics[current_metric][key.strip()] = float(value.strip())
 
    return {m: v["mean"] for m, v in metrics.items()}
 
 
def plot_ae_comparison(
    results_folder,
    model_names,
    experiment_name,
    splitname,
    n_patients,
    latdim_list,
    metric="R2",
    xscale="log",
    save_path=None,
):
    """
    Plot 1 — Compare AE architectures against each other.
 
    For each architecture:
      - solid line  : validation set metric vs latent dim
      - dashed line : train set metric vs latent dim
 
    Parameters
    ----------
    results_folder : Path or str
        Path to the autoencoder results folder (TEMPODATA_FOLDER / "autoencoder").
    model_names : list of str
        e.g. ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"]
    experiment_name : str
        e.g. "lr1e-5"
    splitname : str
        e.g. "split0"
    n_patients : int
        Number of train patients (used to build simulation_name).
    latdim_list : list of int
        Latent dimensions to plot.
    metric : str
        Metric to plot. Default "R2".
    xscale : str
        "log" or "linear".
    save_path : Path or str or None
        If None, saves to RESULTS_FOLDER.
    """
    results_folder = Path(results_folder)
 
    color_map = {
        "AE3dCurrent": "orange",
        "AE3dFCDeep":  "green",
        "AE3dConv":    "blue",
        "AE3dLinear":  "purple",
    }
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for model_name in model_names:
        color = color_map.get(model_name, "gray")
        val_values, train_values, dims_val, dims_train = [], [], [], []
 
        for latent_dim in latdim_list:
            simulation_name = f"{model_name}_{n_patients}patients_{splitname}_{latent_dim}dims"
 
            # Find best_epoch from existing .pth file
            model_dir = results_folder / simulation_name / experiment_name
            # pth_files = sorted(model_dir.glob("_best_*epochs.pth")) if model_dir.exists() else []
            pth_files = sorted(
                                                    model_dir.glob("_best_*epochs.pth"),
                                                    key=lambda p: int(p.stem.replace("_best_", "").replace("epochs", ""))
                                                ) if model_dir.exists() else []
            if len(pth_files) == 0:
                continue
            best_epoch = int(pth_files[-1].stem.replace("_best_", "").replace("epochs", ""))
 
            val_m = _load_summarymetrics(results_folder, simulation_name, experiment_name, best_epoch, "validation")
            train_m = _load_summarymetrics(results_folder, simulation_name, experiment_name, best_epoch, "train")
 
            if val_m is not None and metric in val_m:
                val_values.append(val_m[metric])
                dims_val.append(latent_dim)
            if train_m is not None and metric in train_m:
                train_values.append(train_m[metric])
                dims_train.append(latent_dim)
 
        if dims_val:
            ax.plot(dims_val, val_values, color=color, linewidth=2,
                    marker="o", markersize=5, label=f"{model_name} (val)")
        if dims_train:
            ax.plot(dims_train, train_values, color=color, linewidth=1,
                    linestyle="--", marker="o", markersize=3, label=f"{model_name} (train)")
 
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(metric)
    ax.set_xscale(xscale)
    ax.set_xticks(dims_val)
    ax.set_xticklabels([str(d) for d in dims_val])
    ax.set_title(
        f"AE comparison | {metric} | val set\n"
        f"{n_patients} train patients | {splitname} | {experiment_name}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
 
    if save_path is None:
        save_path = RESULTS_FOLDER / f"AEcomparison_{metric}_{splitname}_{n_patients}patients_{experiment_name}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
 
 
def plot_ae_vs_pca(
    ae_results_folder,
    pca_results_folder,
    model_names,
    experiment_name,
    pca_name,
    splitname,
    n_patients,
    latdim_list_ae,
    latdim_list_pca,
    metric="R2",
    xscale="log",
    save_path=None,
):
    """
    Plot 2 — Compare best AE architectures vs PCA. Informative only —
    do NOT use this plot to select the AE model (use plot_ae_comparison instead).
 
    For each AE architecture and PCA:
      - solid line  : test set metric vs latent dim
      - dashed line : train set metric vs latent dim
 
    PCA is plotted in black.
 
    Parameters
    ----------
    ae_results_folder : Path or str
        Path to TEMPODATA_FOLDER / "autoencoder".
    pca_results_folder : Path or str
        Path to TEMPODATA_FOLDER / "pca_allpatients_res".
    model_names : list of str
        AE architectures to plot.
    experiment_name : str
        AE experiment name, e.g. "lr1e-5".
    pca_name : str
        PCA simulation name, e.g. "PCA_100patients_split0_ED".
    splitname : str
    n_patients : int
        Number of AE train patients.
    latdim_list_ae : list of int
        Latent dims for AE.
    latdim_list_pca : list of int
        Latent dims for PCA (can be denser, e.g. range(1, 100)).
    metric : str
        Metric to plot. Default "R2".
    xscale : str
        "log" or "linear".
    save_path : Path or str or None
    """
    ae_results_folder  = Path(ae_results_folder)
    pca_results_folder = Path(pca_results_folder)
 
    color_map = {
        "AE3dCurrent": "orange",
        "AE3dFCDeep":  "green",
        "AE3dConv":    "blue",
        "AE3dLinear":  "purple",
        "PCA":         "black",
    }
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    # ── AE curves ─────────────────────────────────────────────────────────────
    for model_name in model_names:
        color = color_map.get(model_name, "gray")
        test_values, train_values, dims_test, dims_train = [], [], [], []
 
        for latent_dim in latdim_list_ae:
            simulation_name = f"{model_name}_{n_patients}patients_{splitname}_{latent_dim}dims"
 
            model_dir = ae_results_folder / simulation_name / experiment_name
            # pth_files = sorted(model_dir.glob("_best_*epochs.pth")) if model_dir.exists() else []
            pth_files = sorted(
                                                    model_dir.glob("_best_*epochs.pth"),
                                                    key=lambda p: int(p.stem.replace("_best_", "").replace("epochs", ""))
                                                ) if model_dir.exists() else []
            if len(pth_files) == 0:
                continue
            best_epoch = int(pth_files[-1].stem.replace("_best_", "").replace("epochs", ""))
 
            test_m  = _load_summarymetrics(ae_results_folder, simulation_name, experiment_name, best_epoch, "test")
            train_m = _load_summarymetrics(ae_results_folder, simulation_name, experiment_name, best_epoch, "train")
 
            if test_m is not None and metric in test_m:
                test_values.append(test_m[metric])
                dims_test.append(latent_dim)
            if train_m is not None and metric in train_m:
                train_values.append(train_m[metric])
                dims_train.append(latent_dim)
 
        if dims_test:
            ax.plot(dims_test, test_values, color=color, linewidth=2,
                    marker="o", markersize=5, label=f"{model_name} (test)")
        if dims_train:
            ax.plot(dims_train, train_values, color=color, linewidth=1,
                    linestyle="--", marker="o", markersize=3, label=f"{model_name} (train)")
 
    # ── PCA curve ─────────────────────────────────────────────────────────────
    pca_test_values, pca_train_values, pca_dims_test, pca_dims_train = [], [], [], []
 
    for latent_dim in latdim_list_pca:
        experiment_dim = f"{latent_dim}dims"
 
        test_m  = _load_summarymetrics(pca_results_folder, pca_name, experiment_dim, None, "test")
        train_m = _load_summarymetrics(pca_results_folder, pca_name, experiment_dim, None, "train")
 
        if test_m is not None and metric in test_m:
            pca_test_values.append(test_m[metric])
            pca_dims_test.append(latent_dim)
        if train_m is not None and metric in train_m:
            pca_train_values.append(train_m[metric])
            pca_dims_train.append(latent_dim)
 
    if pca_dims_test:
        ax.plot(pca_dims_test, pca_test_values, color="black", linewidth=2,
                label="PCA (test)")
    if pca_dims_train:
        ax.plot(pca_dims_train, pca_train_values, color="black", linewidth=1,
                linestyle="--", label="PCA (train)")
 
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(metric)
    ax.set_xscale(xscale)
    ax.set_xticks([1]+latdim_list_ae)
    ax.set_xticklabels([str(d) for d in [1]+latdim_list_ae])
    ax.set_title(
        f"AE vs PCA | {metric} | test set (informative only)\n"
        f"{n_patients} train patients | {splitname} | {experiment_name}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
 
    if save_path is None:
        save_path = RESULTS_FOLDER / f"AEvsPC_{metric}_{splitname}_{n_patients}patients_{experiment_name}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_ae_npatients_comparison(
    results_folder,
    model_names,
    experiment_name,
    splitname,
    n_patients_list,
    latdim_lists,
    metrics_dataset="validation",
    metric="R2",
    xscale="log",
    save_path=None,
):
    """
    Compare AE architectures trained on different numbers of patients.
 
    Style encoding:
      - Color     → architecture (orange=AE3dCurrent, green=AE3dFCDeep, blue=AE3dConv)
      - Linewidth → n_patients: thin (1.5) for fewer patients, thick (3) for more
      - Marker    → n_patients: 'o' for fewer patients, 's' for more
 
    Parameters
    ----------
    results_folder : Path or str
        Path to TEMPODATA_FOLDER / "autoencoder".
    model_names : list of str
        AE architectures to plot, e.g. ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"].
    experiment_name : str
        e.g. "baseline"
    splitname : str
        e.g. "split0"
    n_patients_list : list of int
        e.g. [100, 200] — must be sorted ascending.
    latdim_lists : dict
        {n_patients: [list of latent dims]}
        e.g. {100: [4, 8, ..., 100], 200: [4, 8, ..., 200]}
    metrics_dataset : str
        "train", "validation", or "test".
    metric : str
        Metric to plot. Default "R2".
    xscale : str
        "log" or "linear".
    save_path : Path or str or None
    """
    results_folder = Path(results_folder)
 
    color_map = {
        "AE3dCurrent": "orange",
        "AE3dFCDeep":  "green",
        "AE3dConv":    "blue",
        "AE3dLinear":  "purple",
    }
 
    # Style by n_patients rank (thinnest = fewest patients)
    n_sorted = sorted(n_patients_list)
    linewidth_map = {n: 1 + 1.* i for i, n in enumerate(n_sorted)}
    marker_map    = {n: ["o", "s", "^", "D"][i % 4] for i, n in enumerate(n_sorted)}
    linestyle_map = {100: "dotted", 200: "solid"}

    fig, ax = plt.subplots(figsize=(11, 6))
 
    for model_name in model_names:
        color = color_map.get(model_name, "gray")
 
        for n_patients in n_patients_list:
            latdim_list = latdim_lists[n_patients]
            lw     = linewidth_map[n_patients]
            marker = marker_map[n_patients]
            linestyle = linestyle_map[n_patients]

            values, dims = [], []
 
            for latent_dim in latdim_list:
                simulation_name = f"{model_name}_{n_patients}patients_{splitname}_{latent_dim}dims"
                model_dir = results_folder / simulation_name / experiment_name
                pth_files = sorted(
                    model_dir.glob("_best_*epochs.pth"),
                    key=lambda p: int(p.stem.replace("_best_", "").replace("epochs", ""))
                ) if model_dir.exists() else []
                if len(pth_files) == 0:
                    continue
                best_epoch = int(pth_files[-1].stem.replace("_best_", "").replace("epochs", ""))
 
                m = _load_summarymetrics(
                    results_folder, simulation_name, experiment_name, best_epoch, metrics_dataset
                )
                if m is not None and metric in m:
                    values.append(m[metric])
                    dims.append(latent_dim)
 
            if dims:
                ax.plot(
                    dims, values,
                    color=color,
                    linewidth=lw,
                    linestyle = linestyle,
                    marker=marker,
                    markersize=5,
                    label=f"{model_name} ({n_patients} patients)",
                )
 
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(metric)
    ax.set_xscale(xscale)
    ax.set_title(
        f"AE comparison — effect of n_patients | {metric} | {metrics_dataset} set\n"
        f"{splitname} | {experiment_name}"
    )
    ax.set_xticks(sorted(set(d for dims in latdim_lists.values() for d in dims)))
    ax.set_xticklabels([str(d) for d in sorted(set(d for dims in latdim_lists.values() for d in dims))],
                       rotation=45, fontsize=8)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
 
    if save_path is None:
        patients_str = "_".join(str(n) for n in n_patients_list)
        save_path = RESULTS_FOLDER / (
            f"AEnpatients_comparison_{metric}_{metrics_dataset}_{splitname}_{patients_str}patients_{experiment_name}.png"
        )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
