# src/models/regression.py
"""
Data loading, split management, logistic and linear regression models
for PCA-based cardiac group classification and metadata prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
import joblib
import hashlib

from src.config import TEMPODATA_FOLDER
from src.models import pca_spatial as pcs
from src.data import importdata as ipd


def splitname_to_seed(splitname, modulo=2**32):
    digest = hashlib.sha256(splitname.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def get_or_create_split_indices(n_samples, y, splitname, test_size=50, save_folder=None):
    """
    Load a previously saved split if it exists, otherwise create it from the
    split name, save it, and return train/test indices.
    """
    if save_folder is None:
        save_folder = TEMPODATA_FOLDER

    save_folder.mkdir(parents=True, exist_ok=True)

    split_path = save_folder / f"{splitname}_splitindices.joblib"

    if split_path.exists():
        split_data = joblib.load(split_path)

        train_idx = np.asarray(split_data["train_idx"], dtype=int)
        test_idx = np.asarray(split_data["test_idx"], dtype=int)

        if len(train_idx) + len(test_idx) != n_samples:
            raise ValueError(
                f"Saved split '{splitname}' is incompatible with current dataset size: "
                f"{len(train_idx)} + {len(test_idx)} != {n_samples}"
            )

        return train_idx, test_idx

    seed = splitname_to_seed(splitname)

    all_idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=y,
        random_state=seed,
        shuffle=True,
    )

    split_data = {
        "splitname": splitname,
        "seed": seed,
        "n_samples": int(n_samples),
        "train_idx": np.asarray(train_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
    }

    joblib.dump(split_data, split_path, compress=3)

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def load_xy(source_folder, X_folder, savepca_folder, pca_description, maskYN, maskbinYN, imageROIonlyYN, whichtoreturn, group_binYN, group_binvalue, ntraining=100, recalculateXbase=False, defaultsplit=True, splitname=None):
    """
    Load image data (X) and patient metadata (Y), and split into train/test sets.
    """
    # Load X (patient images or mask)
    X = pcs.get_vectorsarray(source_folder, X_folder, details_str=pca_description, mask=maskYN, binary_mask=maskbinYN, image_roi_only=imageROIonlyYN, recalculate=recalculateXbase)
    # Load Y (patient metadata)
    all_files = ipd.import_patientmetapaths(printinfos=False)
    Y = pcs.patient_metalists(all_files, returnonlyone=True, whichtoreturn=whichtoreturn)
    if group_binYN:
        Y = [int(val == group_binvalue) for val in Y]
    Y = np.asarray(Y)
    # Split
    if defaultsplit:
        X_train, X_test = X[:ntraining], X[ntraining:]
        Y_train, Y_test = np.asarray(Y[:ntraining]), np.asarray(Y[ntraining:])
    else:
        if splitname is None:
            raise ValueError("splitname must be provided when defaultsplit=False")
        split_folder = TEMPODATA_FOLDER / savepca_folder
        train_idx, test_idx = get_or_create_split_indices(n_samples=len(X), y=Y, splitname=splitname, test_size=50, save_folder=split_folder)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
    return X_train, X_test, Y_train, Y_test


def n_pc_for_variance(pca, threshold):
    """
    Return the number of PCs needed to reach a given cumulative explained variance threshold.
    """
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    idx = np.where(cumulative >= threshold)[0]
    return int(len(cumulative)) if len(idx) == 0 else int(idx[0] + 1)


def logisticreg(X_train_pca, Y_train, pca_folder, pca_description, Y_name, group_binvalue, n_pc_tokeep, multi_class, recalculateLOGI=True, save=False):
    """
    Train or load a logistic regression classifier (binary or multiclass).
    """
    if multi_class:
        mtc_savestring = "_mtc"
        groupname = ""
        max_iter = 30000
    else:
        mtc_savestring = ""
        groupname = group_binvalue
        max_iter = 10000

    if recalculateLOGI:
        clf = LogisticRegression(max_iter=max_iter, random_state=42, solver="newton-cg")
        clf.fit(X_train_pca, Y_train)
        if save:
            joblib.dump(clf, TEMPODATA_FOLDER / pca_folder + "/" + pca_description + "_" + Y_name + groupname + "_" + repr(n_pc_tokeep) + "pc" + mtc_savestring + "_clf.joblib", compress=3)
    else:
        clf = joblib.load(TEMPODATA_FOLDER / pca_folder + "/" + pca_description + "_" + Y_name + groupname + "_" + repr(n_pc_tokeep) + "pc" + mtc_savestring + "_clf.joblib")
    return clf


def logistic_predictions_results(pca, clf, X_test_pca, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, group_binYN, group_binvalue, addstring=""):
    """
    Compute binary logistic regression metrics and save them to a txt file.
    """
    Y_pred = clf.predict(X_test_pca)
    Y_prob = clf.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_prob)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    cm = confusion_matrix(Y_test, Y_pred)

    results = {
        "PCA_description": pca_description,
        "n_components_kept": n_pc_tokeep,
        "Y_name": Y_name,
        "Group_binary_YN": group_binYN,
        "Group_binary_chosen": group_binvalue,
        "explained_variance_kept": np.sum(pca.explained_variance_ratio_),
        "accuracy": acc,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }

    with open(TEMPODATA_FOLDER / pca_folder + "/" + pca_description + "_" + Y_name + group_binvalue + "_" + repr(n_pc_tokeep) + "pc_predictionresults" + addstring + ".txt", "w") as f:
        f.write("\n========================================\n")
        f.write(f"PCA description: {pca_description}\n")
        f.write(f"Target variable: {Y_name}\n")
        f.write(f"Group_binary_YN: {group_binYN}\n")
        f.write(f"Group value chosen to be =1: {group_binvalue}\n")
        f.write(f"Number of PCs: {n_pc_tokeep}\n")
        f.write(f"Explained variance kept: {np.sum(pca.explained_variance_ratio_[:n_pc_tokeep]):.6f}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"ROC AUC: {auc:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n")

    return results


def logistic_predictions_results_mtc(pca, clf, X_test_pca, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, splitname=""):
    """
    Compute multiclass logistic regression metrics and save them to a txt file.
    """
    Y_pred = clf.predict(X_test_pca)

    acc = accuracy_score(Y_test, Y_pred)
    precision_macro = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(Y_test, Y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)

    precision_per_class = precision_score(Y_test, Y_pred, average=None, labels=clf.classes_, zero_division=0)
    recall_per_class = recall_score(Y_test, Y_pred, average=None, labels=clf.classes_, zero_division=0)

    explained_variance_kept = np.sum(pca.explained_variance_ratio_[:n_pc_tokeep])

    results = {
        "PCA_description": pca_description,
        "n_components_kept": n_pc_tokeep,
        "Y_name": Y_name,
        "explained_variance_kept": explained_variance_kept,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "classes": [str(c) for c in clf.classes_],
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "confusion_matrix": cm,
    }

    with open(TEMPODATA_FOLDER / pca_folder + "/" + pca_description + "_" + splitname + "_" + Y_name + "_" + repr(n_pc_tokeep) + "pc_mtc_predictionresults.txt", "w") as f:
        f.write("\n========================================\n")
        f.write(f"PCA description: {pca_description}\n")
        f.write(f"Target variable: {Y_name}\n")
        f.write(f"Number of PCs: {n_pc_tokeep}\n")
        f.write(f"Explained variance kept: {explained_variance_kept:.6f}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"Precision macro: {precision_macro:.3f}\n")
        f.write(f"Recall macro: {recall_macro:.3f}\n")
        classes_clean = [str(c) for c in clf.classes_]
        f.write(f"Classes: {classes_clean}\n")
        f.write("Precision per class:\n")
        for cls, val in zip(clf.classes_, precision_per_class):
            f.write(f"{str(cls)}: {val:.3f}\n")
        f.write("Recall per class:\n")
        for cls, val in zip(clf.classes_, recall_per_class):
            f.write(f"{str(cls)}: {val:.3f}\n")
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n")

    return results


def linearreg(X_train_pca, Y_train, pca_folder, pca_description, Y_name, n_pc_tokeep, recalculateLIN=True, save=False):
    """
    Train or load a linear regression model.
    """
    filename = (
        TEMPODATA_FOLDER / pca_folder + "/" + pca_description + "_"
        + Y_name + "_" + repr(n_pc_tokeep) + "pc_reg.joblib"
    )
    if recalculateLIN:
        reg = LinearRegression()
        reg.fit(X_train_pca, Y_train)
        if save:
            joblib.dump(reg, filename, compress=3)
    else:
        reg = joblib.load(filename)

    return reg


def linear_predictions_results(pca, reg, X_test_pca, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name):
    """
    Compute linear regression evaluation metrics and save them to a txt file.
    """
    Y_pred = reg.predict(X_test_pca)

    r2 = r2_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)

    explained_variance_kept = np.sum(pca.explained_variance_ratio_[:n_pc_tokeep])

    results = {
        "PCA_description": pca_description,
        "n_components_kept": n_pc_tokeep,
        "Y_name": Y_name,
        "explained_variance_kept": explained_variance_kept,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }

    with open(
        TEMPODATA_FOLDER / pca_folder + "/" + pca_description + "_" + Y_name + "_" + repr(n_pc_tokeep) + "pc_regressionresults.txt",
        "w"
    ) as f:
        f.write("\n========================================\n")
        f.write(f"PCA description: {pca_description}\n")
        f.write(f"Target variable: {Y_name}\n")
        f.write(f"Number of PCs: {n_pc_tokeep}\n")
        f.write(f"Explained variance kept: {explained_variance_kept:.6f}\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")

    return results