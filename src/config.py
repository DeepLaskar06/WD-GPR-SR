"""
Global Configuration and Deterministic Seeding for WD-GPR-SR.
Author: Moonis Laskar (Roll No: 2401CS29)
"""

import os
import random
from pathlib import Path
import numpy as np

# Suppress Matplotlib writable directory warnings by redirecting cache locally
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Project Metadata
AUTHOR = "Moonis Laskar"
ROLL_NO = "2401CS29"
PROJECT_TITLE = "WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction"

# Reproducibility & Seeding
SEEDS = [42, 100, 2026]
PRIMARY_SEED = 42

def set_global_seed(seed: int = PRIMARY_SEED):
    """
    Sets deterministic seeds across Python, NumPy, PyTorch, and OS environment.
    Enforces deterministic algorithm execution for full experimental reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass

    return seed

# Dataset Constants
NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
NASA_NOMINAL_CAP = 2.0  # Rated nominal capacity in Ah (18650 LiCoO2)

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
CALCE_NOMINAL_CAP = 1.1  # Rated nominal capacity in Ah (Prismatic LiCoO2)

# File System Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
SRC_DIR = BASE_DIR / "src"
DATA_RAW_NASA_DIR = BASE_DIR / "data" / "raw" / "nasa"
DATA_RAW_CALCE_DIR = BASE_DIR / "data" / "raw" / "calce"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_FIG_DIR = BASE_DIR / "results" / "figures"
RESULTS_TAB_DIR = BASE_DIR / "results" / "tables"
JETSON_DIR = BASE_DIR / "jetson"

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
