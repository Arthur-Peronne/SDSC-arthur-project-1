# scripts/run_pca_spatial.py
"""
Spatial PCA across patients.

Two use cases:
  1. Standalone PCA study: explained variance, eigenvectors, metadata correlations.
  2. AE comparison: reconstruction metrics (R2, MSE...) on train / val / test sets,
     using the same splits as run_autoencoder.py for a fair comparison.

Supports ED only, ES only, or ED+ES combined datasets.
All output files include a frame tag (ED / ES / ED+ES) to avoid overwriting.
"""

import numpy as np

from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER
from src.models import pca_spatial as pcs
from src.models import pca as pc
from src.training import ae_training as aet
from src.visualization import ae_plots as aep

# ── User choices : DATA ───────────────────────────────────────────────────────
source_folder = "registered_frames"
n_development = 120                 # train + validation patients
n_validation = 20                   # validation patients
splitname = "split0"
recalculateX = False

# Frame selection
use_both_frames = True             # True → ED+ES (300 patients), False → ED only
frame_type = "ED"                   # "ED" or "ES" — used only if use_both_frames=False

# ── User choices : PCA ────────────────────────────────────────────────────────
X_folder = "X_vectors"
pca_folder = "pca_allpatients_res"
imageROIonly = True
maskYS = False
maskbin = False
original_shape = (128, 128, 32)
max_pc_calc = 300
recalculatePCA = False

# ── User choices : RECONSTRUCTION METRICS ────────────────────────────────────
compute_metrics = True
latdim_list_pca = list(range(1, 200))  # 1 to 99 or 1 to 199

# ── User choices : STANDALONE PLOTS ──────────────────────────────────────────
plot_explained_variance = False
plot_pc_values          = False
plot_metadata           = False
plot_eigenvectors_flag  = False     # slow — set True only when needed

pc_n1, pc_n2 = 0, 1                # PC indices for 2D plots
eigenvectors_toplot = 10

# ── User choices : RECONSTRUCTION ───────────────────────────────────────────── 
plot_reconstruction    = False      # set True to reconstruct and plot selected patients
latent_dim_plot = 120         # latent dim to use for reconstruction
recons_auto = True            # Automatically choose 3 patients good/medium/bad reconstruct in the train/val/test
patients_torecons_manual = [(30,"ES"), (110, "ED"),(130, "ED")] # else manual choice

# ── Derived parameters ────────────────────────────────────────────────────────
n_train = n_development - n_validation
n_train_images = n_train * 2 if use_both_frames else n_train
n_val_images = n_validation * 2 if use_both_frames else n_validation

frame_tag = "ED+ES" if use_both_frames else frame_type  # "ED", "ES", ou "ED+ES"
save_suffix = ""
if maskYS: save_suffix += "_gt"
if maskbin: save_suffix += "_bin"
if imageROIonly: save_suffix += "_imgROIonly"

# ── Load data ─────────────────────────────────────────────────────────────────
def _load_X(ft):
    return pcs.get_vectorsarray(
        source_folder,
        X_folder,
        # details_str=pca_description,
        mask=maskYS,
        binary_mask=maskbin,
        image_roi_only=imageROIonly,
        recalculate=recalculateX,
        frame_type=ft,
        flatten=True,
    )

if use_both_frames:
    X_ED = _load_X("ED")
    X_ES = _load_X("ES")
    if X_ED.shape != X_ES.shape:
        raise  ValueError(f"Shape mismatch: X_ED={X_ED.shape} vs X_ES={X_ES.shape}")
    # split + concatenate
    X_train = np.concatenate([X_ED[:n_train], X_ES[:n_train]], axis=0)
    X_val   = np.concatenate([X_ED[n_train:n_development], X_ES[n_train:n_development]], axis=0)
    X_test  = np.concatenate([X_ED[n_development:], X_ES[n_development:]], axis=0)
else:
    X = _load_X(frame_type)   # "ED" ou "ES" selon frame_type
    X_train = X[:n_train]
    X_val   = X[n_train:n_development]
    X_test  = X[n_development:]

# Calculate means (to reconstruct images later) and center X_train, X_val, X_test by row (same as done during pca_patients)
if not maskbin:
    row_means_train = X_train.mean(axis=1, keepdims=True)
    row_means_val   = X_val.mean(axis=1,  keepdims=True)   # before centering
    row_means_test  = X_test.mean(axis=1, keepdims=True)
    X_train -= row_means_train
    X_val  -= row_means_val
    X_test -= row_means_test
else:
    row_means_train = np.zeros((X_train.shape[0], 1))
    row_means_val  = np.zeros((X_val.shape[0],  1))
    row_means_test = np.zeros((X_test.shape[0], 1))


# ── PCA training (on train set only) ─────────────────────────────────────────
pca_name = f"PCA_{n_train_images}patients_{splitname}_{frame_tag}"

pca, X_train_pca, meta = pcs.pca_patients(
    X_train,
    pca_folder,
    pca_name,
    normalize_rows=False, # No need to center, already centered for security before
    recalculatePCA=recalculatePCA,
    max_pc_calc=max_pc_calc,
)

# Project val and test into PCA space
X_val_pca  = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# ── Reconstruction metrics ────────────────────────────────────────────────────
if compute_metrics:

    for metrics_dataset, X_flat, X_pca_sub, offset in [
        ("train",      X_train, X_train_pca, 0),
        ("validation", X_val,   X_val_pca,   n_train_images),
        ("test",       X_test,  X_test_pca,  n_train_images + n_val_images),
    ]:
        for latent_dimensions in latdim_list_pca:
            pcs.pca_compute_metrics(
                X_flat=X_flat,
                X_pca=X_pca_sub,
                pca=pca,
                latent_dimensions=latent_dimensions,
                offset=offset,
                pca_name=pca_name,
                metrics_dataset=metrics_dataset,
                original_shape=original_shape,
                pca_folder=pca_folder,
            )

# ── Standalone PCA plots ──────────────────────────────────────────────────────
plot_tag = f"allpatients{save_suffix}_{frame_tag}"

if plot_explained_variance:
    pc.plot_pca_explipower(pca, plot_tag)

if plot_pc_values:
    for n1, n2 in [(pc_n1, pc_n2), (2, 3), (4, 5), (6, 7), (8, 9)]:
        pc.plot_pcvalues_2d(
            X_train_pca, n1, n2,
            plot_tag,
            "_pc_in_eigenbase",
            scale_str="Patient number",
            segments=False,
            axisscale_fixed=False,
        )

if plot_metadata:
    for n1, n2 in [(pc_n1, pc_n2), (2, 3), (4, 5), (6, 7), (8, 9)]:
        pcs.plot_pca_patientmeta(X_train_pca, n1, n2)

if plot_eigenvectors_flag:
    pcs.plot_eigenvectors(
        X_train,
        pca,
        original_shape,
        save_suffix,
        eigenvectors_toplot=min(eigenvectors_toplot, max_pc_calc),
    )

# ── PCA reconstruction plots  ──────────────────────────────────────────────────────
if plot_reconstruction:
 
    # Calculate means to re-center images in plot
    row_means_train = meta["row_means"]   # shape (n_train_images, 1)
    if not maskbin:
        row_means_val   = X_val.mean(axis=1,  keepdims=True)   # before centering
        row_means_test  = X_test.mean(axis=1, keepdims=True)
    else:
        row_means_val  = np.zeros((X_val.shape[0],  1))
        row_means_test = np.zeros((X_test.shape[0], 1))
    
    # Select the right split for auto selection
    X_flat_plot, X_pca_plot, offset_plot = X_test, X_test_pca, n_train_images + n_val_images
 
    # Recompute metrics for latent_dim_plot on all patients of the split
    all_metrics_plot = []
    for i, x_patient_flat in enumerate(X_flat_plot):
        x_recon_flat = (
            X_pca_plot[i, :latent_dim_plot]
            @ pca.components_[:latent_dim_plot, :]
            + pca.mean_
        )
        x_patient_3d = x_patient_flat.reshape(original_shape)
        x_recon_3d   = x_recon_flat.reshape(original_shape)
 
        metrics = aet.reconstruction_metrics(
            x_true=x_patient_3d,
            x_pred=x_recon_3d,
            patient_number=offset_plot + 1 + i,
            simulation_name=pca_name,
            n_epochs=None,
            metrics_dataset="test",
            savemetrics=False,
        )
        all_metrics_plot.append(metrics)
 
    # Select patients to reconstruct
    if recons_auto:
        selected = aep.ae_select_representative_patients(
            all_metrics_plot,
            use_both_frames=use_both_frames,
            n_train_images=n_train_images,
            n_val_images=n_val_images,
            n_development=n_development,
        )
        patients_torecons = [(v["real_patient"], v["frame_type"]) for v in selected.values()]
    else:
        patients_torecons = patients_torecons_manual

    # Plot
    aep.pca_plotcompare_selected(
        patients_torecons=patients_torecons,
        X_train_pca=X_train_pca,
        X_val_pca=X_val_pca,
        X_test_pca=X_test_pca,
        pca=pca,
        latent_dimensions=latent_dim_plot,
        original_shape=original_shape,
        use_both_frames=use_both_frames,
        n_development=n_development,
        n_train_images=n_train_images,
        n_val_images=n_val_images,
        split_name=splitname,
        row_means_train=row_means_train,
        row_means_val=row_means_val,
        row_means_test=row_means_test,
    )
