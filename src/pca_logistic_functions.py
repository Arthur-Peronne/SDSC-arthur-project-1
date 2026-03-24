# src/pca_logistic_functions.py
"""
Functions for the functions for the PCA "each patient" and followed by the logistic regression with patient metadata
"""

import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
import joblib
import os
import re
import glob
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import hashlib

from paths import * 
import pca_eachpatient_functions as pef 
import importdata_functions as idf


def splitname_to_seed(splitname, modulo=2**32):
    digest = hashlib.sha256(splitname.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo

def get_or_create_split_indices(n_samples, y, splitname, test_size=50, save_folder=None):
    """
    Load a previously saved split if it exists, otherwise create it from the
    split name, save it, and return train/test indices.
    """
    if save_folder is None:
        save_folder = path_tempodata_folder

    os.makedirs(save_folder, exist_ok=True)

    split_path = os.path.join(save_folder, f"{splitname}_splitindices.joblib")

    if os.path.exists(split_path):
        split_data = joblib.load(split_path)

        train_idx = np.asarray(split_data["train_idx"], dtype=int)
        test_idx = np.asarray(split_data["test_idx"], dtype=int)

        # Optional safety checks
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

def load_xy(source_folder, X_folder, savepca_folder, pca_description, maskYN, maskbinYN, imageROIonlyYN, whichtoreturn, group_binYN, group_binvalue, ntraining = 100, recalculateXbase= False, defaultsplit = True, splitname = None):
    """
    """
    # Load X (patient images or mask) 
    X = pef.get_vectorsarray(source_folder, X_folder, details_str = pca_description, mask=maskYN, binary_mask=maskbinYN, image_roi_only=imageROIonlyYN, recalculate= recalculateXbase)
    # Load Y (patient data)
    all_files = idf.import_patientmetapaths(printinfos=False)
    Y  = pef.patient_metalists(all_files, returnonlyone=True, whichtoreturn=whichtoreturn)
    if group_binYN:
        Y = [int(val == group_binvalue) for val in Y]
    Y = np.asarray(Y)
    # Split
    if defaultsplit:
        X_train, X_test = X[:ntraining], X[ntraining:]
        Y_train, Y_test = np.asarray(Y[:ntraining]), np.asarray(Y[ntraining:])
    else:
        if splitname is None:
            raise ValueError("splitname must be provided when defaultsplit=False")
        split_folder = os.path.join(path_tempodata_folder, savepca_folder)
        train_idx, test_idx = get_or_create_split_indices(n_samples=len(X), y=Y, splitname=splitname, test_size=50, save_folder=split_folder)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # seed = splitname_to_seed(splitname)
        # X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size = 50, stratify=Y, random_state = seed)
        # Ytrain, Y_test = np.asarray(Y_train, Y_test)
    return X_train, X_test, Y_train, Y_test

def n_pc_for_variance(pca, threshold):
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    idx = np.where(cumulative >= threshold)[0]
    return int(len(cumulative)) if len(idx) == 0 else int(idx[0] + 1)

def logisticreg(X_train_pca, Y_train, pca_folder, pca_description, Y_name, group_binvalue, n_pc_tokeep, multi_class, recalculateLOGI=True, save = False):
    """
    """
    if multi_class:
        mtc_savestring = "_mtc"
        groupname = ""
        max_iter = 30000
    else:
        mtc_savestring = ""
        groupname = group_binvalue
        max_iter = 10000

    if recalculateLOGI: # Recalculate and save
        clf = LogisticRegression(max_iter=max_iter, random_state=42, solver="newton-cg")
        clf.fit(X_train_pca, Y_train)
        if save:
            joblib.dump(clf, path_tempodata_folder + pca_folder + "/" + pca_description + "_"+ Y_name + groupname + "_"  + repr(n_pc_tokeep) + "pc" + mtc_savestring + "_clf.joblib", compress=3)
    else: # or load
        clf = joblib.load(path_tempodata_folder + pca_folder + "/" + pca_description + "_"+ Y_name + groupname + "_"  + repr(n_pc_tokeep)  + "pc" + mtc_savestring + "_clf.joblib")
    return clf

def logistic_predictions_results(pca, clf, X_test_pca, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, group_binYN, group_binvalue, addstring=""):
    """
    """
    # Predictions
    Y_pred = clf.predict(X_test_pca)
    Y_prob = clf.predict_proba(X_test_pca)[:, 1]
    # Results
    acc = accuracy_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_prob)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    cm = confusion_matrix(Y_test, Y_pred)
    # Results storage
    results = {
            "PCA_description": pca_description ,
            "n_components_kept": n_pc_tokeep,
            "Y_name" : Y_name,
            "Group_binary_YN": group_binYN,
            "Group_binary_chosen": group_binvalue,
            "explained_variance_kept": np.sum(pca.explained_variance_ratio_),
            "accuracy": acc,
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
                        }
    # Save results to file
    with open(path_tempodata_folder + pca_folder + "/" + pca_description + "_"+ Y_name + group_binvalue + "_"  + repr(n_pc_tokeep) + "pc_predictionresults" + addstring + ".txt", "w") as f:
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

def logistic_predictions_results_mtc(pca, clf, X_test_pca, Y_test, pca_description, pca_folder, n_pc_tokeep, Y_name, splitname =""):
    """
    """
    # Predictions
    Y_pred = clf.predict(X_test_pca)

    # Metrics
    acc = accuracy_score(Y_test, Y_pred)
    precision_macro = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(Y_test, Y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)

    # Per-class metrics
    precision_per_class = precision_score(Y_test, Y_pred, average=None, labels=clf.classes_, zero_division=0)
    recall_per_class = recall_score(Y_test, Y_pred, average=None, labels=clf.classes_, zero_division=0)

    explained_variance_kept = np.sum(pca.explained_variance_ratio_[:n_pc_tokeep])

    # Results storage
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

    # Save results to file
    with open(path_tempodata_folder + pca_folder + "/" + pca_description + "_" + splitname + "_" + Y_name + "_" + repr(n_pc_tokeep) + "pc_mtc_predictionresults.txt", "w") as f:
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

def parse_prediction_result_file(filepath):
    """
    Read one *_predictionresults.txt file and extract metrics.
    """
    results = {}

    # Extract number of PCs from filename, e.g. "..._24pc_predictionresults.txt"
    filename = os.path.basename(filepath)
    match = re.search(r'_(\d+)pc_predictionresults\.txt$', filename)
    if match is None:
        raise ValueError(f"Could not extract number of PCs from filename: {filename}")
    results["n_pc"] = int(match.group(1))

    with open(filepath, "r") as f:
        text = f.read()

    patterns = {
        "accuracy": r"Accuracy:\s*([0-9.]+)",
        "roc_auc": r"ROC AUC:\s*([0-9.]+)",
        "precision": r"Precision:\s*([0-9.]+)",
        "recall": r"Recall:\s*([0-9.]+)",
        "explained_variance_kept": r"Explained variance kept:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m is None:
            raise ValueError(f"Could not find '{key}' in file: {filename}")
        results[key] = float(m.group(1))

    return results

def plot_prediction_results(pca_folder, pca_description,Y_name, group_binvalue, addstring = ""):
    """
    Parameters
    ----------
    results_folder : str
        Folder containing the txt result files.
    prefix : str
        Common prefix of files, e.g. 'REGvoxROI_groupDCM'
    outname : str
        Output png filename.
    """
    results_folder = os.path.join(path_tempodata_folder, pca_folder + "/")
    pattern = os.path.join(results_folder, f"{pca_description}_{Y_name}{group_binvalue}_*pc_predictionresults.txt")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    all_results = []
    for filepath in files:
        try:
            all_results.append(parse_prediction_result_file(filepath))
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if len(all_results) == 0:
        raise ValueError("No valid result files could be parsed.")

    # Sort by number of PCs
    all_results.sort(key=lambda d: d["n_pc"])

    n_pc = [d["n_pc"] for d in all_results]
    accuracy = [d["accuracy"] for d in all_results]
    roc_auc = [d["roc_auc"] for d in all_results]
    precision = [d["precision"] for d in all_results]
    recall = [d["recall"] for d in all_results]
    explained_variance_kept = [d["explained_variance_kept"] for d in all_results]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)

    # Plot 1: AUC + Accuracy
    ax1.plot(n_pc, roc_auc, marker="o", label="ROC AUC")
    ax1.plot(n_pc, accuracy, marker="o", label="Accuracy")
    ax1.plot(n_pc, explained_variance_kept, marker="o", linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
    ax1.set_xscale("log")
    ax1.set_xticks(n_pc)
    ax1.set_xticklabels([str(x) for x in n_pc])
    ax1.set_ylabel("Score")
    ax1.set_title(f"{pca_description}_{Y_name}{group_binvalue} - Overall performance")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Recall + Precision
    ax2.plot(n_pc, recall, marker="o", label="Recall")
    ax2.plot(n_pc, precision, marker="o", label="Precision")
    ax2.plot(n_pc, explained_variance_kept, marker="o", linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
    ax2.set_xscale("log")
    ax2.set_xticks(n_pc)
    ax2.set_xticklabels([str(x) for x in n_pc])
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{pca_description}_{Y_name}{group_binvalue} - Positive class behavior")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # outpath = os.path.join(results_folder, outname)
    outpath = path_resultsfolder + pca_description + "_"+ Y_name + group_binvalue + "_" + "predictionresults" + addstring + " .png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved plot to: {outpath}")

def parse_prediction_result_file_mtc(filepath):
    """
    Read one *pc_mtc_predictionresults.txt file and extract multiclass metrics.
    """
    results = {}

    filename = os.path.basename(filepath)
    match = re.search(r'_(\d+)pc_mtc_predictionresults(?:_.*)?\.txt$', filename)
    if match is None:
        raise ValueError(f"Could not extract number of PCs from filename: {filename}")
    results["n_pc"] = int(match.group(1))

    with open(filepath, "r") as f:
        text = f.read()

    # Global metrics
    patterns = {
        "accuracy": r"Accuracy:\s*([0-9.]+)",
        "precision_macro": r"Precision macro:\s*([0-9.]+)",
        "recall_macro": r"Recall macro:\s*([0-9.]+)",
        "explained_variance_kept": r"Explained variance kept:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m is None:
            raise ValueError(f"Could not find '{key}' in file: {filename}")
        results[key] = float(m.group(1))

    # Classes
    m = re.search(r"Classes:\s*\[(.*?)\]", text)
    if m is None:
        raise ValueError(f"Could not find 'Classes' in file: {filename}")

    classes_str = m.group(1)
    classes = []
    for c in classes_str.split(","):
        c = c.strip()
        if c.startswith("np.str_(") and c.endswith(")"):
            c = c[len("np.str_("):-1].strip()
        c = c.strip("'").strip('"')
        classes.append(c)
    results["classes"] = classes

    # Precision per class block
    m = re.search(r"Precision per class:\n(.*?)Recall per class:", text, re.S)
    if m is None:
        raise ValueError(f"Could not find 'Precision per class' block in file: {filename}")
    precision_block = m.group(1)

    precision_per_class = {}
    for line in precision_block.strip().splitlines():
        cls, val = line.split(":")
        precision_per_class[cls.strip()] = float(val.strip())
    results["precision_per_class"] = precision_per_class

    # Recall per class block
    m = re.search(r"Recall per class:\n(.*?)Confusion matrix:", text, re.S)
    if m is None:
        raise ValueError(f"Could not find 'Recall per class' block in file: {filename}")
    recall_block = m.group(1)

    recall_per_class = {}
    for line in recall_block.strip().splitlines():
        cls, val = line.split(":")
        recall_per_class[cls.strip()] = float(val.strip())
    results["recall_per_class"] = recall_per_class

    return results

def plot_prediction_results_mtc(pca_folder, pca_description, Y_name, splitname = ""):
    """
    Plot multiclass logistic regression results from saved txt files.
    """
    results_folder = os.path.join(path_tempodata_folder, pca_folder + "/")
    pattern = os.path.join(results_folder, f"{pca_description}_{splitname}_{Y_name}_*pc_mtc_predictionresults.txt")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    all_results = []
    for filepath in files:
        try:
            all_results.append(parse_prediction_result_file_mtc(filepath))
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if len(all_results) == 0:
        raise ValueError("No valid multiclass result files could be parsed.")

    # Sort by number of PCs
    all_results.sort(key=lambda d: d["n_pc"])

    n_pc = [d["n_pc"] for d in all_results]
    accuracy = [d["accuracy"] for d in all_results]
    precision_macro = [d["precision_macro"] for d in all_results]
    recall_macro = [d["recall_macro"] for d in all_results]
    explained_variance_kept = [d["explained_variance_kept"] for d in all_results]

    classes = all_results[0]["classes"]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(9, 14), sharex=True)

    # Plot 1: Accuracy
    ax1.plot(n_pc, accuracy, marker="o", label="Accuracy")
    ax1.plot(
        n_pc, explained_variance_kept,
        linestyle="--", color="gray", alpha=0.35,
        label="Cum. explained variance"
    )
    ax1.set_xscale("log")
    ax1.set_xticks(n_pc)
    ax1.set_xticklabels([str(x) for x in n_pc])
    ax1.set_ylabel("Score")
    ax1.set_title(f"{pca_description}_{Y_name} - Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Recall per class
    for cls in classes:
        values = [d["recall_per_class"][cls] for d in all_results]
        ax2.plot(n_pc, values, marker="o", label=f"Recall {cls}")
    ax2.plot(
        n_pc, explained_variance_kept,
        linestyle="--", color="gray", alpha=0.35,
        label="Cum. explained variance"
    )
    ax2.set_xscale("log")
    ax2.set_xticks(n_pc)
    ax2.set_xticklabels([str(x) for x in n_pc])
    ax2.set_ylabel("Score")
    ax2.set_title(f"{pca_description}_{Y_name} - Recall per class")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Precision per class
    for cls in classes:
        values = [d["precision_per_class"][cls] for d in all_results]
        ax3.plot(n_pc, values, marker="o", label=f"Precision {cls}")
    ax3.plot(
        n_pc, explained_variance_kept,
        linestyle="--", color="gray", alpha=0.35,
        label="Cum. explained variance"
    )
    ax3.set_xscale("log")
    ax3.set_xticks(n_pc)
    ax3.set_xticklabels([str(x) for x in n_pc])
    ax3.set_xlabel("Number of PCs")
    ax3.set_ylabel("Score")
    ax3.set_title(f"{pca_description}_{Y_name} - Precision per class")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    outpath = os.path.join(
        path_resultsfolder,
        f"{pca_description}_{splitname}_{Y_name}_mtc_predictionresults.png"
    )
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved plot to: {outpath}")

def plot_confusion_matrix_mtc(pca_folder, pca_description, Y_name, n_pc, splitname=""):
    """
    Plot the normalized confusion matrix for a chosen number of PCs
    from the saved multiclass prediction results.
    """

    results_folder = os.path.join(path_tempodata_folder, pca_folder + "/")

    filepath = os.path.join(
        results_folder,
        f"{pca_description}_{splitname}_{Y_name}_{n_pc}pc_mtc_predictionresults.txt"
    )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        text = f.read()

    # -------- Extract classes --------
    match = re.search(r"Classes:\s*\[(.*?)\]", text)
    if match is None:
        raise ValueError("Could not find class list in file")

    classes = [c.strip().strip("'") for c in match.group(1).split(",")]

    # -------- Extract confusion matrix --------
    match = re.search(r"Confusion matrix:\n([\s\S]*?\]\])", text)

    if match is None:
        raise ValueError("Could not find confusion matrix in file")

    cm_str = match.group(1)

    rows = []
    for line in cm_str.strip().split("\n"):
        line = line.replace("[", "").replace("]", "").strip()
        if line:
            rows.append([int(x) for x in line.split()])

    cm = np.array(rows)
    
    # -------- Normalize rows --------
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # -------- Plot heatmap --------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=1
    )

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"{pca_description}_{Y_name} – Confusion matrix ({n_pc} PCs)")
    plt.tight_layout()

    outpath = (
        path_resultsfolder
        + f"{pca_description}_{splitname}_{Y_name}_{n_pc}pc_confusionmatrix.png"
    )

    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {outpath}")

def linearreg(X_train_pca, Y_train, pca_folder, pca_description, Y_name, n_pc_tokeep, recalculateLIN=True, save = False):
    """
    Train or load a linear regression model.
    """
    filename = (
        path_tempodata_folder + pca_folder + "/" + pca_description + "_"
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
    # Predictions
    Y_pred = reg.predict(X_test_pca)

    # Metrics
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
        path_tempodata_folder + pca_folder + "/" + pca_description + "_" + Y_name + "_" + repr(n_pc_tokeep) + "pc_regressionresults.txt",
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

def parse_regression_result_file(filepath):
    """
    Read one *pc_regressionresults.txt file and extract regression metrics.
    """
    results = {}

    filename = os.path.basename(filepath)
    match = re.search(r'_(\d+)pc_regressionresults\.txt$', filename)
    if match is None:
        raise ValueError(f"Could not extract number of PCs from filename: {filename}")
    results["n_pc"] = int(match.group(1))

    with open(filepath, "r") as f:
        text = f.read()

    patterns = {
        "explained_variance_kept": r"Explained variance kept:\s*([0-9eE+.\-]+)",
        "r2": r"R2:\s*([0-9eE+.\-]+)",
        "rmse": r"RMSE:\s*([0-9eE+.\-]+)",
        "mae": r"MAE:\s*([0-9eE+.\-]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m is None:
            raise ValueError(f"Could not find '{key}' in file: {filename}")
        results[key] = float(m.group(1))

    return results

def plot_regression_results(pca_folder, pca_description, Y_name):
    """
    Plot regression metrics from saved txt files:
    - R2 vs number of PCs
    - RMSE and MAE vs number of PCs
    """
    results_folder = os.path.join(path_tempodata_folder, pca_folder + "/")
    pattern = os.path.join(results_folder, f"{pca_description}_{Y_name}_*pc_regressionresults.txt")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    all_results = []
    for filepath in files:
        try:
            all_results.append(parse_regression_result_file(filepath))
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if len(all_results) == 0:
        raise ValueError("No valid regression result files could be parsed.")

    # Sort by number of PCs
    all_results.sort(key=lambda d: d["n_pc"])

    n_pc = [d["n_pc"] for d in all_results]
    r2 = [d["r2"] for d in all_results]
    rmse = [d["rmse"] for d in all_results]
    mae = [d["mae"] for d in all_results]
    explained_variance_kept = [d["explained_variance_kept"] for d in all_results]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)

    # Plot 1: R2
    ax1.plot(n_pc, r2, marker="o", label="R²")
    ax1.plot(
        n_pc, explained_variance_kept,
        marker="o", linestyle="--", color="gray", alpha=0.35,
        label="Cum. explained variance"
    )
    ax1.set_xscale("log")
    ax1.set_xticks(n_pc)
    ax1.set_xticklabels([str(x) for x in n_pc])
    ax1.set_ylabel("Score")
    ax1.set_title(f"{pca_description}_{Y_name} - R²")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: RMSE + MAE
    ax2.plot(n_pc, rmse, marker="o", label="RMSE")
    ax2.plot(n_pc, mae, marker="o", label="MAE")
    ax2.plot(
        n_pc, explained_variance_kept,
        marker="o", linestyle="--", color="gray", alpha=0.35,
        label="Cum. explained variance"
    )
    ax2.set_xscale("log")
    ax2.set_xticks(n_pc)
    ax2.set_xticklabels([str(x) for x in n_pc])
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel(Y_name)
    ax2.set_title(f"{pca_description}_{Y_name} - Prediction errors")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    outpath = os.path.join(
        path_resultsfolder,
        f"{pca_description}_{Y_name}_regressionresults.png"
    )
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved regression plots to: {outpath}")

def plot_regression_predicted_vs_true(reg, X_test_pca, Y_test, pca_description, Y_name, n_pc_tokeep):
    """
    Plot predicted vs true values for a chosen regression model.
    """
    Y_pred = reg.predict(X_test_pca)

    y_min = min(np.min(Y_test), np.min(Y_pred))
    y_max = max(np.max(Y_test), np.max(Y_pred))

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.8)
    plt.plot([y_min, y_max], [y_min, y_max], linestyle="--", color="gray", alpha=0.7)

    plt.xlabel(f"True {Y_name}")
    plt.ylabel(f"Predicted {Y_name}")
    plt.title(f"{pca_description}_{Y_name} - Predicted vs True ({n_pc_tokeep} PCs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = os.path.join(
        path_resultsfolder,
        f"{pca_description}_{Y_name}_{n_pc_tokeep}pc_predicted_vs_true.png"
    )
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved predicted-vs-true plot to: {outpath}")

def plot_average_confusion_matrix_mtc_simple(filepaths, outpath, title="Average confusion matrix"):
    """
    Load several multiclass result txt files, average their row-normalized
    confusion matrices, and plot the result.

    Parameters
    ----------
    filepaths : list of str
        List of txt result files containing a confusion matrix.
    outpath : str
        Path to save the output figure.
    title : str
        Plot title.
    """
    cms = []
    classes_ref = None

    for filepath in filepaths:
        with open(filepath, "r") as f:
            text = f.read()

        # Extract classes
        match = re.search(r"Classes:\s*\[(.*?)\]", text)
        if match is None:
            raise ValueError(f"Could not find class list in file: {filepath}")
        classes = [c.strip().strip("'").strip('"') for c in match.group(1).split(",")]

        # Extract confusion matrix
        match = re.search(r"Confusion matrix:\n([\s\S]*?\]\])", text)
        if match is None:
            raise ValueError(f"Could not find confusion matrix in file: {filepath}")

        cm_str = match.group(1)

        rows = []
        for line in cm_str.strip().split("\n"):
            line = line.replace("[", "").replace("]", "").strip()
            if line:
                rows.append([int(x) for x in line.split()])

        cm = np.array(rows, dtype=float)

        # Normalize rows
        cm = cm / cm.sum(axis=1, keepdims=True)

        if classes_ref is None:
            classes_ref = classes
        elif classes != classes_ref:
            raise ValueError(
                f"Class order mismatch in file {filepath}\n"
                f"Expected: {classes_ref}\n"
                f"Found: {classes}"
            )

        cms.append(cm)

    # Average matrices
    cm_avg = np.mean(cms, axis=0)

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_avg,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes_ref,
        yticklabels=classes_ref,
        vmin=0,
        vmax=1
    )

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved average confusion matrix to: {outpath}")

def get_mtc_result_files(results_folder):
    """
    Return all multiclass prediction result files in a folder.

    Parameters
    ----------
    results_folder : str
        Folder containing the txt result files.

    Returns
    -------
    list of str
        Sorted list of filepaths.
    """
    pattern = os.path.join(results_folder, "*mtc_predictionresults*.txt")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No mtc_predictionresults files found in: {results_folder}")

    return sorted(files)