# scripts/run_comparison_aepca.py
"""
Comparison plots between AE architectures and PCA.
 
Plot 1 — AE vs AE (validation set):
    Compare AE architectures against each other.
    Use this plot to select the best AE architecture and latent dim.
 
Plot 2 — AE vs PCA (test set, informative only):
    Compare best AE architectures against PCA.
    Do NOT use this plot to select the AE model — use Plot 1 for that.
 
Plot 3 — AE n_patients comparison:
    Compare AE architectures trained on different numbers of patients.
    Helps assess the effect of data augmentation (e.g. ED vs ED+ES).
"""
 
from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER
from src.training import ae_training as aet
from src.visualization import ae_plots as aep

# ── User choices : WHICH PLOTS TO PRODUCE ────────────────────────────────────
run_plot1_ae_comparison        = True   # AE vs AE, val set
run_plot2_ae_vs_pca                 = True   # AE vs PCA, test set (informative)
run_plot3_npatients                 = False   # AE 100 vs 200 patients comparison
run_plot4_hyper_comparison =  True # AE hyperparameter trainings comparison (optuna/baseline)
run_stats_nepochs                      = True 

# ── User choices : DATA ───────────────────────────────────────────────────────
splitname = "split0"
 
# ── User choices : AE (shared) ────────────────────────────────────────────────
model_names    = ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"] # ["AE3dCurrent", "AE3dFCDeep", "AE3dConv"]
experiment_name = "optuna"
 
# ── User choices : PLOT 1 & 2 (single n_patients study) ──────────────────────
frame_tag      = "ED+ES"            # "ED" or "ED+ES"
n_patients     = 200                # n_train_images for this study
latdim_list_ae = [4, 8, 12, 20, 28, 40, 60, 88, 120, 160, 200] # [4, 8, 12, 20, 28, 40, 60, 80, 100] or [4, 8, 12, 20, 28, 40, 60, 88, 120, 160]  ADD 200 later
 
# ── User choices : PLOT 2 — PCA ───────────────────────────────────────────────
pca_name       = f"PCA_{n_patients}patients_{splitname}_{frame_tag}"
latdim_list_pca = list(range(1, 200))
 
# ── User choices : PLOT 3 — n_patients comparison ────────────────────────────
n_patients_list = [100, 200]
latdim_lists = {
    100: [4, 8, 12, 20, 28, 40, 60, 80, 100],
    200: [4, 8, 12, 20, 28, 40, 60, 88, 120, 160, 200], # ADD 200 later
}
plot3_metrics_dataset = "validation"   # "train", "validation", or "test"
 
# ── User choices : PLOTS (shared) ────────────────────────────────────────────
metric = "R2"
xscale = "log"
 
# ── Paths ─────────────────────────────────────────────────────────────────────
ae_results_folder  = TEMPODATA_FOLDER / "autoencoder"
pca_results_folder = TEMPODATA_FOLDER / "pca_allpatients_res"
 
# ── Plot 1 — AE vs AE (validation set) ───────────────────────────────────────
if run_plot1_ae_comparison:
    aep.plot_ae_comparison(
        results_folder=ae_results_folder,
        model_names=model_names,
        experiment_name=experiment_name,
        splitname=splitname,
        n_patients=n_patients,
        latdim_list=latdim_list_ae,
        metric=metric,
        xscale=xscale,
    )
 
# ── Plot 2 — AE vs PCA (test set, informative only) ──────────────────────────
if run_plot2_ae_vs_pca:
    aep.plot_ae_vs_pca(
        ae_results_folder=ae_results_folder,
        pca_results_folder=pca_results_folder,
        model_names=model_names,
        experiment_name=experiment_name,
        pca_name=pca_name,
        splitname=splitname,
        n_patients=n_patients,
        latdim_list_ae=latdim_list_ae,
        latdim_list_pca=latdim_list_pca,
        metric=metric,
        xscale=xscale,
    )
 
# ── Plot 3 — AE n_patients comparison ────────────────────────────────────────
if run_plot3_npatients:
    aep.plot_ae_npatients_comparison(
        results_folder=ae_results_folder,
        model_names=model_names,
        experiment_name=experiment_name,
        splitname=splitname,
        n_patients_list=n_patients_list,
        latdim_lists=latdim_lists,
        metrics_dataset=plot3_metrics_dataset,
        metric=metric,
        xscale=xscale,
    )

# ── Plot 4 — AE hyperparameter trainings comparison ────────────────────────────────────────
if run_plot4_hyper_comparison:
    aep.plot_ae_experiment_comparison(
        results_folder=TEMPODATA_FOLDER / "autoencoder",
        model_names=model_names,
        experiment_names=["baseline", "optuna"],
        splitname=splitname,
        n_patients=200,
        latdim_list=latdim_list_ae,
        metrics_dataset="validation",
        metric=metric,
        xscale=xscale,
    )

# Stats on n_epochs from ae_training early stopping 

if run_stats_nepochs:
    stats = aet.get_best_epochs_stats(
        results_folder=TEMPODATA_FOLDER / "autoencoder",
        model_name="AE3dFCDeep",
        n_patients_list=[n_patients],
        splitname=splitname,
        latdim_list=latdim_list_ae,
        experiment_name=experiment_name,
    )
    aet.print_and_save_best_epochs_stats(stats,     model_name = "AE3dFCDeep", experiment_name = experiment_name)