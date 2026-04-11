# src/config.py
"""
Central configuration — paths loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Environment-dependent paths (defined in .env)
DATADIR = Path(os.environ["DATADIR"])
RESULTS_FOLDER = Path(os.environ["RESULTS_FOLDER"])
TEMPODATA_FOLDER = Path(os.environ["TEMPODATA_FOLDER"])