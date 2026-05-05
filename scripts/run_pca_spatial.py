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

# ── User choices : DATA ───────────────────────────────────────────────────────
source_folder = "registered_frames"
n_development = 120                 # train + validation patients
n_validation = 20                   # validation patients
splitname = "split0"
recalculateX = True

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
recalculatePCA = True

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

# ── Derived parameters ────────────────────────────────────────────────────────
n_train = n_development - n_validation
n_train_images = n_train * 2 if use_both_frames else n_train
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
        raise ValueError(...)
    # split + concatenate
    X_train = np.concatenate([X_ED[:n_train], X_ES[:n_train]], axis=0)
    X_val   = np.concatenate([X_ED[n_train:n_development], X_ES[n_train:n_development]], axis=0)
    X_test  = np.concatenate([X_ED[n_development:], X_ES[n_development:]], axis=0)
else:
    X = _load_X(frame_type)   # "ED" ou "ES" selon frame_type
    X_train = X[:n_train]
    X_val   = X[n_train:n_development]
    X_test  = X[n_development:]

print(f"Frame tag : {frame_tag}")
print(f"Train     : {X_train.shape[0]} patients")
print(f"Val       : {X_val.shape[0]} patients")
print(f"Test      : {X_test.shape[0]} patients")

# ── PCA training (on train set only) ─────────────────────────────────────────
pca_name = f"PCA_{n_train_images}patients_{splitname}_{frame_tag}"

pca, X_train_pca, meta = pcs.pca_patients(
    X_train,
    pca_folder,
    pca_name,
    normalize_rows=not maskbin,
    recalculatePCA=recalculatePCA,
    max_pc_calc=max_pc_calc,
)

# Project val and test into PCA space
# Apply same per-patient centering as during training if normalize_rows=True
if not maskbin:
    X_val_pca  = pca.transform(X_val  - X_val.mean(axis=1,  keepdims=True))
    X_test_pca = pca.transform(X_test - X_test.mean(axis=1, keepdims=True))
else:
    X_val_pca  = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

# ── Reconstruction metrics ────────────────────────────────────────────────────
if compute_metrics:
    for latent_dimensions in latdim_list_pca:

        for metrics_dataset, X_flat, X_pca_sub, offset in [
            ("train",      X_train, X_train_pca, 0),
            ("validation", X_val,   X_val_pca,   n_train),
            ("test",       X_test,  X_test_pca,  n_development),
        ]:
            all_metrics = []

            for i, x_patient_flat in enumerate(X_flat):
                x_recon_flat = (
                    X_pca_sub[i, :latent_dimensions]
                    @ pca.components_[:latent_dimensions, :]
                    + pca.mean_
                )
                x_patient_3d = x_patient_flat.reshape(original_shape)
                x_recon_3d   = x_recon_flat.reshape(original_shape)

                metrics = aet.reconstruction_metrics(
                    x_true=x_patient_3d,
                    x_pred=x_recon_3d,
                    patient_number=offset + 1 + i,
                    simulation_name=pca_name,
                    n_epochs=None,
                    metrics_dataset=metrics_dataset,
                    savemetrics=False,
                )
                all_metrics.append(metrics)

            aet.ae_aggregate_metrics(
                all_metrics,
                simulation_name=pca_name,
                experiment_name= f"{latent_dimensions}dims",
                n_epochs=None,
                metrics_dataset=metrics_dataset,
                ae = False,
            )
            print(
                f"[PCA {latent_dimensions}dims | {metrics_dataset}] "
                f"R2 mean = {np.mean([m['R2'] for m in all_metrics]):.4f}"
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