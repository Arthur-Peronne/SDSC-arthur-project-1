# src/visualization/regression_plots.py
"""
Result parsing and plotting functions for logistic and linear regression
on PCA-based cardiac representations.
"""

import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import TEMPODATA_FOLDER, RESULTS_FOLDER


def parse_prediction_result_file(filepath):
    """
    Read one *_predictionresults.txt file and extract binary classification metrics.
    """
    results = {}

    filename = Path(filepath).name
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


def plot_prediction_results(pca_folder, pca_description, Y_name, group_binvalue, addstring=""):
    """
    Plot binary logistic regression metrics (AUC, accuracy, precision, recall)
    vs number of PCs, from saved txt result files.
    """
    results_folder = TEMPODATA_FOLDER / pca_folder
    pattern = str(results_folder / f"{pca_description}_{Y_name}{group_binvalue}_*pc_predictionresults.txt")
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

    all_results.sort(key=lambda d: d["n_pc"])

    n_pc = [d["n_pc"] for d in all_results]
    accuracy = [d["accuracy"] for d in all_results]
    roc_auc = [d["roc_auc"] for d in all_results]
    precision = [d["precision"] for d in all_results]
    recall = [d["recall"] for d in all_results]
    explained_variance_kept = [d["explained_variance_kept"] for d in all_results]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)

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

    outpath = RESULTS_FOLDER / f"{pca_description}_{Y_name}{group_binvalue}_predictionresults{addstring}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved plot to: {outpath}")


def parse_prediction_result_file_mtc(filepath):
    """
    Read one *pc_mtc_predictionresults.txt file and extract multiclass metrics.
    """
    results = {}

    filename = Path(filepath).name
    match = re.search(r'_(\d+)pc_mtc_predictionresults(?:_.*)?\.txt$', filename)
    if match is None:
        raise ValueError(f"Could not extract number of PCs from filename: {filename}")
    results["n_pc"] = int(match.group(1))

    with open(filepath, "r") as f:
        text = f.read()

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

    m = re.search(r"Precision per class:\n(.*?)Recall per class:", text, re.S)
    if m is None:
        raise ValueError(f"Could not find 'Precision per class' block in file: {filename}")
    precision_block = m.group(1)

    precision_per_class = {}
    for line in precision_block.strip().splitlines():
        cls, val = line.split(":")
        precision_per_class[cls.strip()] = float(val.strip())
    results["precision_per_class"] = precision_per_class

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


def plot_prediction_results_mtc(pca_folder, pca_description, Y_name, splitname=""):
    """
    Plot multiclass logistic regression metrics (accuracy, recall, precision per class)
    vs number of PCs, from saved txt result files.
    """
    results_folder = TEMPODATA_FOLDER / pca_folder
    pattern = str(results_folder / f"{pca_description}_{splitname}_{Y_name}_*pc_mtc_predictionresults.txt")
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

    all_results.sort(key=lambda d: d["n_pc"])

    n_pc = [d["n_pc"] for d in all_results]
    accuracy = [d["accuracy"] for d in all_results]
    explained_variance_kept = [d["explained_variance_kept"] for d in all_results]
    classes = all_results[0]["classes"]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(9, 14), sharex=True)

    ax1.plot(n_pc, accuracy, marker="o", label="Accuracy")
    ax1.plot(n_pc, explained_variance_kept, linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
    ax1.set_xscale("log")
    ax1.set_xticks(n_pc)
    ax1.set_xticklabels([str(x) for x in n_pc])
    ax1.set_ylabel("Score")
    ax1.set_title(f"{pca_description}_{Y_name} - Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for cls in classes:
        values = [d["recall_per_class"][cls] for d in all_results]
        ax2.plot(n_pc, values, marker="o", label=f"Recall {cls}")
    ax2.plot(n_pc, explained_variance_kept, linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
    ax2.set_xscale("log")
    ax2.set_xticks(n_pc)
    ax2.set_xticklabels([str(x) for x in n_pc])
    ax2.set_ylabel("Score")
    ax2.set_title(f"{pca_description}_{Y_name} - Recall per class")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    for cls in classes:
        values = [d["precision_per_class"][cls] for d in all_results]
        ax3.plot(n_pc, values, marker="o", label=f"Precision {cls}")
    ax3.plot(n_pc, explained_variance_kept, linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
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

    outpath = RESULTS_FOLDER / f"{pca_description}_{splitname}_{Y_name}_mtc_predictionresults.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved plot to: {outpath}")


def plot_confusion_matrix_mtc(pca_folder, pca_description, Y_name, n_pc, splitname=""):
    """
    Plot the normalized confusion matrix for a chosen number of PCs
    from the saved multiclass prediction results.
    """
    results_folder = TEMPODATA_FOLDER / pca_folder
    filepath = results_folder / f"{pca_description}_{splitname}_{Y_name}_{n_pc}pc_mtc_predictionresults.txt"

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        text = f.read()

    match = re.search(r"Classes:\s*\[(.*?)\]", text)
    if match is None:
        raise ValueError("Could not find class list in file")
    classes = [c.strip().strip("'") for c in match.group(1).split(",")]

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
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

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

    outpath = RESULTS_FOLDER / f"{pca_description}_{splitname}_{Y_name}_{n_pc}pc_confusionmatrix.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {outpath}")


def parse_regression_result_file(filepath):
    """
    Read one *pc_regressionresults.txt file and extract regression metrics.
    """
    results = {}

    filename = Path(filepath).name
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
    Plot regression metrics (R2, RMSE, MAE) vs number of PCs,
    from saved txt result files.
    """
    results_folder = TEMPODATA_FOLDER / pca_folder
    pattern = str(results_folder / f"{pca_description}_{Y_name}_*pc_regressionresults.txt")
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

    all_results.sort(key=lambda d: d["n_pc"])

    n_pc = [d["n_pc"] for d in all_results]
    r2 = [d["r2"] for d in all_results]
    rmse = [d["rmse"] for d in all_results]
    mae = [d["mae"] for d in all_results]
    explained_variance_kept = [d["explained_variance_kept"] for d in all_results]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)

    ax1.plot(n_pc, r2, marker="o", label="R²")
    ax1.plot(n_pc, explained_variance_kept, marker="o", linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
    ax1.set_xscale("log")
    ax1.set_xticks(n_pc)
    ax1.set_xticklabels([str(x) for x in n_pc])
    ax1.set_ylabel("Score")
    ax1.set_title(f"{pca_description}_{Y_name} - R²")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(n_pc, rmse, marker="o", label="RMSE")
    ax2.plot(n_pc, mae, marker="o", label="MAE")
    ax2.plot(n_pc, explained_variance_kept, marker="o", linestyle="--", color="gray", alpha=0.35, label="Cum. explained variance")
    ax2.set_xscale("log")
    ax2.set_xticks(n_pc)
    ax2.set_xticklabels([str(x) for x in n_pc])
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel(Y_name)
    ax2.set_title(f"{pca_description}_{Y_name} - Prediction errors")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    outpath = RESULTS_FOLDER / f"{pca_description}_{Y_name}_regressionresults.png"
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

    outpath = RESULTS_FOLDER / f"{pca_description}_{Y_name}_{n_pc_tokeep}pc_predicted_vs_true.png"
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
    outpath : str or Path
        Path to save the output figure.
    title : str
        Plot title.
    """
    cms = []
    classes_ref = None

    for filepath in filepaths:
        with open(filepath, "r") as f:
            text = f.read()

        match = re.search(r"Classes:\s*\[(.*?)\]", text)
        if match is None:
            raise ValueError(f"Could not find class list in file: {filepath}")
        classes = [c.strip().strip("'").strip('"') for c in match.group(1).split(",")]

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

    cm_avg = np.mean(cms, axis=0)

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
    results_folder : str or Path
        Folder containing the txt result files.

    Returns
    -------
    list of str
        Sorted list of filepaths.
    """
    pattern = str(Path(results_folder) / "*mtc_predictionresults*.txt")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No mtc_predictionresults files found in: {results_folder}")

    return sorted(files)