# scripts/run_ae_hyperparam.py
"""
Optuna hyperparameter optimisation for AE3dFCDeep.

Each trial trains AE3dFCDeep with early stopping and returns
the best validation loss. Results are persisted in SQLite so
the study can be interrupted and resumed at any time.

Usage:
    python scripts/run_ae_hyperparam.py
"""

import optuna
from src.config import TEMPODATA_FOLDER
from src.training import ae_optuna as aeo 

# ── User choices ──────────────────────────────────────────────────────────────
N_TRIALS        = 40
MODEL_NAME      = "AE3dFCDeep"
LATENT_DIM      = 120          # for example
N_EPOCHS        = 500
BATCH_SIZE      = 1
SPLITNAME       = "split0"
USE_BOTH_FRAMES = True
N_DEVELOPMENT   = 120
N_VALIDATION    = 20
STUDY_NAME      = "AE3dFCDeep_regularisation_v1"
DB_PATH         = TEMPODATA_FOLDER / "optuna" / f"{STUDY_NAME}.db"

# Reduce noise in the logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

if __name__ == "__main__":
    config = {
        "n_trials":        N_TRIALS,
        "model_name":      MODEL_NAME,
        "latent_dim":      LATENT_DIM,
        "n_epochs":        N_EPOCHS,
        "batch_size":      BATCH_SIZE,
        "splitname":       SPLITNAME,
        "use_both_frames": USE_BOTH_FRAMES,
        "n_development":   N_DEVELOPMENT,
        "n_validation":    N_VALIDATION,
        "study_name":      STUDY_NAME,
        "db_path":         DB_PATH,
    }
    study = aeo.run_optuna(config)