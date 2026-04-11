# scripts/pca_logistic_script.py
"""
Script for the functions for the PCA "each patient" and followed by the logistic regression with patient metadata
"""

import numpy as np 

from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER
from src.models import regression as reg
from src.visualization import regression_plots as rgp
from src.models import pca_spatial as pcs

 
# PARAMETERS
# 1 / Global parameters
Y_name = "group" # "group", "height", "weight"
group_binYN = False
group_binvalue = "RV" # 'DCM', 'HCM', "MINF", "NOR", "RV"
# 0 / Pca parameters -> which PCA to do
source_folder = "registered_framesBIS"
pca_description = "REGvoxROI"
maskYN, maskbinYN = False, False
imageROIonlyYN = True
pca_folder = "pca_allpatients_res"
cumvar_threshold = 1.0 # 1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.4
cumvar_threshold_list = [2.0, 0.99, 0.98, 0.965, 0.95, 0.92, 0.9, 0.87, 0.85, 0.82, 0.78, 0.74, 0.7, 0.68, 0.65, 0.5, 0.4] # For PCA on voxROI images
# cumvar_threshold_list = [2.0, 0.99, 0.97, 0.94, 0.9, 0.85, 0.8, 0.73, 0.64, 0.6, 0.56, 0.53, 0.5, 0.45, 0.4, 0.36, 0.3, 0.2] # For PCA on binary mask images
n_idealpc_confusion = 12 
# n_idealpc_regression = 7
splitname = "split5"

# # Load data: X and Y
# X_train, X_test, Y_train, Y_test  = reg.load_xy(source_folder, "X_vectors", pca_folder, pca_description, maskYN, maskbinYN, imageROIonlyYN, Y_name, group_binYN, group_binvalue, recalculateXbase= False, defaultsplit = False, splitname = splitname)

# # PCA 
# pca, X_train_pca, meta = pcs.pca_patients(X_train, pca_folder, pca_description, normalize_rows=not maskbinYN, recalculatePCA = False, addstring= "_" + splitname)
# X_test_pca = pca.transform(X_test)
# n_pc_tokeep = reg.n_pc_for_variance(pca, cumvar_threshold)
# X_train_pca_n = np.asarray(X_train_pca[:, :n_pc_tokeep], dtype=np.float64, order="C") # Convert into float64, helps the logistic regression solver
# X_test_pca_n  = np.asarray(X_test_pca[:, :n_pc_tokeep], dtype=np.float64, order="C")

# # Logistic regression 
# clf = reg.logisticreg(X_train_pca_n, Y_train, pca_folder, pca_description, Y_name, group_binvalue, n_pc_tokeep, multi_class = not group_binYN, recalculateLOGI=True)
# if group_binYN: # binary regression
#     results = reg.logistic_predictions_results(pca, clf, X_test_pca_n, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, group_binYN, group_binvalue)
# else: # multiclass regression 
#     results = reg.logistic_predictions_results_mtc(pca, clf, X_test_pca_n, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name)
# Linear regression 
# reg = reg.linearreg(X_train_pca_n, Y_train, pca_folder, pca_description, Y_name, n_pc_tokeep, recalculateLIN=True)
# results = reg.linear_predictions_results(pca, reg, X_test_pca_n, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name)

# # Do all pipeline with different cumvar thresholds
# X_train, X_test, Y_train, Y_test  = reg.load_xy(source_folder, "X_vectors", pca_folder, pca_description, maskYN, maskbinYN, imageROIonlyYN, Y_name, group_binYN, group_binvalue, recalculateXbase= False, defaultsplit = False, splitname = splitname)
# pca, X_train_pca, meta = pcs.pca_patients(X_train, pca_folder, pca_description, normalize_rows=not maskbinYN, recalculatePCA = True, addstring= "_" + splitname)
# if not maskbinYN:
#     X_test = X_test - X_test.mean(axis=1, keepdims=True)
# X_test_pca = pca.transform(X_test)
# for cumvar_threshold in cumvar_threshold_list:
#     n_pc_tokeep = reg.n_pc_for_variance(pca, cumvar_threshold)
#     X_train_pca_n = np.asarray(X_train_pca[:, :n_pc_tokeep], dtype=np.float64, order="C") # Convert into float64, helps the logistic regression solver
#     X_test_pca_n  = np.asarray(X_test_pca[:, :n_pc_tokeep], dtype=np.float64, order="C")
#     # Logistic regression
#     clf = reg.logisticreg(X_train_pca_n, Y_train, pca_folder, pca_description, Y_name, group_binvalue, n_pc_tokeep, multi_class = not group_binYN, recalculateLOGI=True)
#     if group_binYN: # binary regression
#         results = reg.logistic_predictions_results(pca, clf, X_test_pca_n, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, group_binYN, group_binvalue)
#     else: # multiclass regression 
#         results = reg.logistic_predictions_results_mtc(pca, clf, X_test_pca_n, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, splitname= splitname)
#     # Linear regression
#     reg = reg.linearreg(X_train_pca_n, Y_train, pca_folder, pca_description, Y_name, n_pc_tokeep, recalculateLIN=True)
#     results = reg.linear_predictions_results(pca, reg, X_test_pca_n, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name)

# # Plot all results
# if group_binYN: # binary regression
#     rgp.plot_prediction_results(pca_folder, pca_description,Y_name, group_binvalue)
# else:
#     rgp.plot_prediction_results_mtc(pca_folder, pca_description, Y_name, splitname= splitname)
#     rgp.plot_confusion_matrix_mtc(pca_folder, pca_description, Y_name, n_idealpc_confusion, splitname= splitname)
# rgp.plot_regression_results(pca_folder, pca_description, Y_name)
# rgp.plot_regression_predicted_vs_true(reg, X_test_pca_n, Y_test, pca_description, Y_name, n_idealpc_regression) # WARNING: change n_pc_tokeep = n_idealpc_regression to get the right X_test_pca_n !!!

# Average confusion matrix
filepaths = reg.get_mtc_result_files(TEMPODATA_FOLDER / pca_folder)
print(filepaths)  # optional check
rgp.plot_average_confusion_matrix_mtc_simple(filepaths=filepaths, outpath=RESULTS_FOLDER / "REGvoxROI_group_avg_confusionmatrix.png")
