"""
config.py
Central configuration for the Colorectal Cancer (CRC) Identification &
Invasion-Oriented Staging project.

Folder structure expected (matches the user's screenshots):

project1/
└── dataset/
    ├── binary/
    │   ├── NOR/   -> normal patches
    │   └── TUM/   -> tumor patches
    └── stage/
        ├── stage0/  -> mucus tissue
        ├── stage1/  -> stroma tissue
        ├── stage2/  -> muscle tissue
        └── stage3/  -> adipose tissue

WSIs (TCGA-COAD / TCGA-READ, 5 each) live separately and are only used for
END-TO-END inference/testing (scripts/infer_wsi.py), not for training.
"""

import os
import torch

# ----------------------------------------------------------------------
# PATHS  (edit these to match where you actually keep the data)
# ----------------------------------------------------------------------
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT   = os.path.join(PROJECT_ROOT, "dataset")

BINARY_DIR     = os.path.join(DATASET_ROOT, "binary")   # NOR / TUM
STAGE_DIR      = os.path.join(DATASET_ROOT, "stage")    # stage0..stage3

WSI_DIR        = os.path.join(PROJECT_ROOT, "wsi_raw")       # raw .svs files
WSI_PATCH_DIR  = os.path.join(PROJECT_ROOT, "wsi_patches")   # tiled output

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")
REPORT_DIR     = os.path.join(PROJECT_ROOT, "reports")

for d in [CHECKPOINT_DIR, LOG_DIR, REPORT_DIR, WSI_PATCH_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------
BINARY_CLASSES = ["NOR", "TUM"]                 # 0 = normal, 1 = tumor
STAGE_CLASSES  = ["stage0", "stage1", "stage2", "stage3"]
STAGE_NAMES    = {
    "stage0": "Mucus tissue (Stage 0 - mucosal)",
    "stage1": "Stroma tissue (Stage 1 - submucosal invasion)",
    "stage2": "Muscle tissue (Stage 2 - muscularis invasion)",
    "stage3": "Adipose tissue (Stage 3 - serosal/pericolic fat invasion)",
}

# ----------------------------------------------------------------------
# IMAGE / PREPROCESSING
# ----------------------------------------------------------------------
PATCH_SIZE      = 224          # model input size
TILE_SIZE_WSI   = 256      # raw tile size cut from WSI before resize
TILE_STRIDE_WSI = 256       # non-overlapping by default
WSI_LEVEL_MAG   = 10           # target magnification (20x) for tiling
TISSUE_THRESHOLD = 0.15        # min fraction of tissue pixels to keep a tile

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Macenko reference stain vectors / max concentrations are fit from a
# reference patch at runtime (see utils/stain_normalization.py) and cached.
MACENKO_REFERENCE_PATCH = os.path.join(PROJECT_ROOT, "reference_patch.png")

# ----------------------------------------------------------------------
# SPLITS
# ----------------------------------------------------------------------
# User requirement: 75% train / 25% test overall.
# A validation slice is still needed for early-stopping / best-checkpoint
# selection during training, so it is carved OUT OF the 75% training pool
# rather than added as a third top-level split:
#   -> 60% actual training, 15% validation (both come from the 75% pool)
#   -> 25% held-out test set (never touched until final evaluation)
TRAIN_SPLIT = 0.60   # of the WHOLE dataset
VAL_SPLIT   = 0.15   # of the WHOLE dataset (carved from the 75% train pool)
TEST_SPLIT  = 0.25   # of the WHOLE dataset (fixed, held out)
SPLIT_SEED  = 42
# NOTE: splitting must be done at slide/patient level where slide IDs are
# recoverable from filenames (see utils/dataset.py: group_key_from_filename)

# ----------------------------------------------------------------------
# TRAINING HYPERPARAMETERS (defaults; Optuna overrides these during tuning)
# ----------------------------------------------------------------------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 32
NUM_EPOCHS      = 40
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.3
EARLY_STOP_PATIENCE = 7
LABEL_SMOOTHING = 0.05
FOCAL_GAMMA     = 2.0          # for class imbalance
NUM_WORKERS     = 4

# Confidence thresholds for "Unidentified -> consult doctor"
STAGE_CONF_THRESHOLD    = 0.55   # min softmax prob to accept a stage prediction
STAGE_ENTROPY_THRESHOLD = 1.0    # max normalized entropy to accept (nats)

# Deepest-invasion-stage aggregation (used by scripts/infer_wsi.py and
# webapp/inference_runner.py -- centralized here so both stay in sync).
# A stage counts as "present" on a slide only if it exceeds
# DEEPEST_STAGE_PRESENT_FRACTION of that slide's tumor patches, with a
# small absolute floor (DEEPEST_STAGE_MIN_COUNT) so a single stray
# misclassified patch can't be reported as "present" on its own.
# IMPORTANT: keep the floor small -- it must never exceed the total tumor
# patch count on small/demo slides, or every stage gets rejected even when
# one is clearly dominant (a real bug: min_count=20 silently produced
# "None" on any slide with fewer than ~20 tumor patches of a given stage).
DEEPEST_STAGE_PRESENT_FRACTION      = 0.02   # 2% of tumor patches
DEEPEST_STAGE_UNIDENTIFIED_FRACTION = 0.05   # 5% of tumor patches -> "and beyond"
DEEPEST_STAGE_MIN_COUNT             = 3

# Optuna
OPTUNA_TRIALS = 25
