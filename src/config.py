"""
config.py — Single source of truth for all project settings.

Modify paths and hyperparameters here. Every other module imports from this file.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR             = PROJECT_ROOT / "data"
REAL_IMAGE_DIR       = DATA_DIR / "images-224" / "images-224"   # Flat folder of NIH 224×224 PNGs
SYNTH_IMAGE_DIR      = DATA_DIR / "images-s"                    # ComfyUI-generated synthetic images
CSV_PATH             = DATA_DIR / "Data_Entry_2017.csv"         # NIH metadata
TRAIN_VAL_LIST_PATH  = DATA_DIR / "train_val_list_NIH.txt"      # Official NIH train/val filenames
TEST_LIST_PATH       = DATA_DIR / "test_list_NIH.txt"           # Official NIH test filenames

# Aliases (used by step scripts)
REAL_DATA_DIR   = REAL_IMAGE_DIR
SYNTH_DATA_DIR  = SYNTH_IMAGE_DIR
METADATA_CSV    = CSV_PATH

# Output directories
OUTPUT_DIR      = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
LOG_DIR         = OUTPUT_DIR / "logs"
PLOT_DIR        = OUTPUT_DIR / "plots"
REPORT_DIR      = OUTPUT_DIR / "reports"

# Create output dirs if they don't exist
for d in [CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2. CLASSES
# ──────────────────────────────────────────────
# Focus on a manageable subset of the 15 NIH labels.
# Adjust this list based on your dataset availability.
TARGET_CLASSES = [
    "Atelectasis",
    "Consolidation",
    "Effusion",
    "No Finding",
]

NUM_CLASSES = len(TARGET_CLASSES)

# Mapping: class name → integer label
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# ──────────────────────────────────────────────
# 3. MODEL
# ──────────────────────────────────────────────
MODEL_NAME  = "google/vit-base-patch16-224-in21k"
IMG_SIZE    = 224

# ──────────────────────────────────────────────
# 4. TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 4
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 1e-4
EPOCHS          = 20
EARLY_STOP_PATIENCE = 5

# Mixed precision (auto-disabled when running on CPU)
USE_AMP = True

# Train / Validation split ratio (applied to NIH train_val_list)
# The test set uses the official NIH test_list
VAL_RATIO = 0.15   # 15% of train_val_list becomes validation

# ──────────────────────────────────────────────
# 5. REPRODUCIBILITY
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# 6. DEVICE
# ──────────────────────────────────────────────
import torch

try:
    import torch_directml
    DEVICE = torch_directml.device()
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
