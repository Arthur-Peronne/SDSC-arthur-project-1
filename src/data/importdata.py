# src/data/importdata.py
"""
Functions to import data
"""

import nibabel as nib
import glob
from pathlib import Path

from src.config import DATADIR, TEMPODATA_FOLDER


def extract_nii_file(datatype_tochoose, patient_name, file_name, print_infos=False):
    """
    Find the nii.gz file to extract, and return the nii object
    """
    path_toextract = DATADIR / datatype_tochoose / patient_name / f"{patient_name}_{file_name}.nii.gz"
    nii_obj = nib.load(path_toextract)
    if print_infos:
        print(path_toextract)
        print(nii_obj.shape)
    return nii_obj


def convert_nii_file(nii_obj):
    """
    Into a Numpy array
    """
    data_array = nii_obj.get_fdata()
    return data_array


def read_info_cfg(filepath):
    """
    Read an ACDC Info.cfg file and return a dictionary of key-value pairs.
    """
    info = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            info[key] = value

    return info


def load_allframes(only01=False):
    """
    Load paths to all frame NIfTI files and their GT masks.
    """
    if only01:
        training_img = glob.glob(str(DATADIR / "training/patient*/patient*_frame01.nii.gz"))
        training_img90 = glob.glob(str(DATADIR / "training/patient*/patient*_frame04.nii.gz"))  # For patient090
        testing_img = glob.glob(str(DATADIR / "testing/patient*/patient*_frame01.nii.gz"))
        all_img = training_img + training_img90 + testing_img
        training_gt = glob.glob(str(DATADIR / "training/patient*/patient*_frame01_gt.nii.gz"))
        training_gt90 = glob.glob(str(DATADIR / "training/patient*/patient*_frame04_gt.nii.gz"))  # For patient090
        testing_gt = glob.glob(str(DATADIR / "testing/patient*/patient*_frame01_gt.nii.gz"))
        all_gt = training_gt + training_gt90 + testing_gt
    else:
        training_img = glob.glob(str(DATADIR / "training/patient*/patient*_frame[0-9][0-9].nii.gz"))
        testing_img = glob.glob(str(DATADIR / "testing/patient*/patient*_frame[0-9][0-9].nii.gz"))
        all_img = training_img + testing_img
        training_gt = glob.glob(str(DATADIR / "training/patient*/patient*_frame*_gt.nii.gz"))
        testing_gt = glob.glob(str(DATADIR / "testing/patient*/patient*_frame*_gt.nii.gz"))
        all_gt = training_gt + testing_gt

    return all_img, all_gt


def load_allframes_resampled(only01=True):
    """
    Load paths to all resampled frame NIfTI files and their GT masks.
    """
    if only01:
        all_img = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame01_resampled.nii.gz"))
        all_gt = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame01_resampled_gt.nii.gz"))
        img_90 = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame04_resampled.nii.gz"))
        gt_90 = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame04_resampled_gt.nii.gz"))
        all_img, all_gt = all_img + img_90, all_gt + gt_90
    else:
        all_img = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame[0-9][0-9]_resampled.nii.gz"))
        all_gt = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame[0-9][0-9]_resampled_gt.nii.gz"))

    return all_img, all_gt

def load_allframes_registered(folder="registered_frames", frame_type="ED"):
    """
    Return sorted paths to registered NIfTI files and their GT masks.
 
    Parameters
    ----------
    folder : str
        Subfolder inside TEMPODATA_FOLDER containing the registered files.
    frame_type : str
        "ED" → ED frame (frame01 for all patients, frame04 for patient090)
        "ES" → ES frame (the other frame for each patient)
 
    Returns
    -------
    all_img : list of str
        Sorted paths to registered image files.
    all_gt : list of str
        Sorted paths to registered GT mask files.
    """
    if frame_type not in {"ED", "ES"}:
        raise ValueError("frame_type must be 'ED' or 'ES'")
 
    base = TEMPODATA_FOLDER / folder
 
    if frame_type == "ED":
        # frame01 for all patients except patient090 (frame04)
        all_img = sorted(glob.glob(str(base / "patient*_frame01_registered.nii.gz")))
        all_gt  = sorted(glob.glob(str(base / "patient*_frame01_registered_gt.nii.gz")))
        img_90  = sorted(glob.glob(str(base / "patient090_frame04_registered.nii.gz")))
        gt_90   = sorted(glob.glob(str(base / "patient090_frame04_registered_gt.nii.gz")))
        all_img = sorted(all_img + img_90)
        all_gt  = sorted(all_gt  + gt_90)
 
    else:  # ES
        # Everything except frame01 and patient090's frame04
        all_img_raw = sorted(glob.glob(str(base / "patient*_frame*_registered.nii.gz")))
        all_gt_raw  = sorted(glob.glob(str(base / "patient*_frame*_registered_gt.nii.gz")))
 
        def _is_ES(path):
            name = Path(path).name
            if "patient090" in name:
                return "_frame04_" not in name   # for patient090, ES = anything but frame04
            else:
                return "_frame01_" not in name   # for all others, ES = anything but frame01
 
        all_img = sorted(p for p in all_img_raw if _is_ES(p))
        all_gt  = sorted(p for p in all_gt_raw  if _is_ES(p))
 
    return all_img, all_gt

def load_allgt_res(onlytraining=False):
    """
    Load paths to all resampled GT mask files.
    """
    if onlytraining:
        all_gt = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient0*_frame[0-9][0-9]_resampled_gt.nii.gz"))
        all_gt += glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient100_frame[0-9][0-9]_resampled_gt.nii.gz"))
    else:
        all_gt = glob.glob(str(TEMPODATA_FOLDER / "resampled_frames/patient*_frame[0-9][0-9]_resampled_gt.nii.gz"))
    return all_gt


def import_patientmetapaths(printinfos=True):
    """
    Return sorted paths to all patient Info.cfg files (training + testing).
    """
    training_files = sorted(glob.glob(str(DATADIR / "training/patient*/Info.cfg")))
    testing_files = sorted(glob.glob(str(DATADIR / "testing/patient*/Info.cfg")))
    all_files = training_files + testing_files
    if printinfos:
        print("First file:", all_files[0])
        print("Last file :", all_files[-1])
        print("Total     :", len(all_files))
    return all_files


def load_allcroppedframes():
    """
    Load paths to all cropped frame NIfTI files.
    """
    all_img_crop = glob.glob(str(TEMPODATA_FOLDER / "cropped_frames/patient*_frame*_cropped.nii.gz"))
    return sorted(all_img_crop)


def get_patient_acdc_path(patient_id, file_type="frame", base_path=None, check_exists=False):
    """
    Return the path to one ACDC file for a given patient.

    Parameters
    ----------
    patient_id : int
        Patient number, from 1 to 150.
    file_type : str
        Type of file to retrieve:
        - "frame" : initial frame NIfTI
        - "mask"  : GT mask of the initial frame
        - "4d"    : 4D NIfTI with all epochs
        Default: "frame"
    base_path : Path or None
        Root folder of the ACDC dataset. Defaults to DATADIR.
    check_exists : bool
        If True, raise FileNotFoundError when the path does not exist.

    Returns
    -------
    Path
        Full path to the requested file.
    """
    if base_path is None:
        base_path = DATADIR

    base_path = Path(base_path)

    if not (1 <= patient_id <= 150):
        raise ValueError("patient_id must be between 1 and 150")

    if file_type not in {"frame", "mask", "4d"}:
        raise ValueError("file_type must be one of: 'frame', 'mask', '4d'")

    subset = "training" if patient_id <= 100 else "testing"
    patient_str = f"patient{patient_id:03d}"

    frame_num = 4 if patient_id == 90 else 1
    frame_str = f"frame{frame_num:02d}"

    if file_type == "frame":
        filename = f"{patient_str}_{frame_str}.nii.gz"
    elif file_type == "mask":
        filename = f"{patient_str}_{frame_str}_gt.nii.gz"
    else:
        filename = f"{patient_str}_4d.nii.gz"

    full_path = base_path / subset / patient_str / filename

    if check_exists and not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    return full_path


def get_patient_modified_path(patient_id, folder, file_type="frame", base_path=None, check_exists=False):
    """
    Return the path to a modified (e.g. registered) NIfTI file for a given patient.

    Parameters
    ----------
    patient_id : int
        Patient number (1 to 150)
    folder : str
        Subfolder inside tempodata (e.g. "registered_framesBIS")
    file_type : str
        Type of file:
        - "frame" : modified image (default)
        - "mask"  : modified GT mask
    base_path : Path or None
        Root path to tempodata. Defaults to TEMPODATA_FOLDER.
    check_exists : bool
        If True, raise error if file does not exist

    Returns
    -------
    Path
        Full path to the requested file.
    """
    if base_path is None:
        base_path = TEMPODATA_FOLDER

    base_path = Path(base_path)

    if not (1 <= patient_id <= 150):
        raise ValueError("patient_id must be between 1 and 150")

    if file_type not in {"frame", "mask"}:
        raise ValueError("file_type must be 'frame' or 'mask'")

    patient_str = f"patient{patient_id:03d}"

    frame_num = 4 if patient_id == 90 else 1
    frame_str = f"frame{frame_num:02d}"

    if file_type == "frame":
        filename = f"{patient_str}_{frame_str}_registered.nii.gz"
    else:
        filename = f"{patient_str}_{frame_str}_registered_gt.nii.gz"

    full_path = base_path / folder / filename

    if check_exists and not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    return full_path