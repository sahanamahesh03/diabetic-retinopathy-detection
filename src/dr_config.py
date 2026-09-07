"""
config.py
----------
Centralized configuration for the Diabetic Retinopathy Detection project.
All paths, hyperparameters, and constants live here so every other script
can import them instead of hardcoding values in multiple places.
"""

import os

# ----------------------------
# BASE PROJECT PATHS
# ----------------------------
# os.path.dirname(__file__) gets the folder this config.py file is in (src/)
# We go one level up (..) to get the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw dataset location (as downloaded/extracted from Kaggle)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, "train.csv")

# Processed data location (resized images we will create in Step 6)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Where we'll save train/val/test split CSVs (Step 6)
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

# Where trained models get saved
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Where output plots (accuracy/loss curves, confusion matrix) get saved
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# Where Grad-CAM heatmap images get saved
GRADCAM_DIR = os.path.join(BASE_DIR, "outputs", "gradcam_results")

# Where training logs go (for TensorBoard, etc.)
LOGS_DIR = os.path.join(BASE_DIR, "logs")


# ----------------------------
# IMAGE SETTINGS
# ----------------------------
# EfficientNetB0 was pretrained on ImageNet using 224x224 images.
# Using the same size lets us take full advantage of transfer learning.
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3  # RGB

# ----------------------------
# CLASS LABELS
# ----------------------------
# The 5 diabetic retinopathy severity classes, in order (0 to 4)
CLASS_NAMES = [
    "No_DR",        # 0
    "Mild",         # 1
    "Moderate",     # 2
    "Severe",       # 3
    "Proliferative" # 4
]
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------
# TRAINING HYPERPARAMETERS
# ----------------------------
# Batch size: how many images the model looks at before updating its weights.
# Kept moderate since we're training on CPU (no GPU) - smaller batches use
# less memory and avoid slowdowns/crashes on limited RAM.
BATCH_SIZE = 16

# Number of full passes through the training data.
# We start conservative since CPU training is slow; we can increase later
# if the model needs more training to improve.
EPOCHS = 20

# Learning rate: how big a step the optimizer takes when updating weights.
# 0.0001 is a common safe starting point for fine-tuning a pretrained model.
LEARNING_RATE = 0.0001

# ----------------------------
# TRAIN / VALIDATION / TEST SPLIT RATIOS
# ----------------------------
# 70% train, 15% validation, 15% test - standard split for a dataset this size
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed - ensures our train/val/test split is reproducible
# (running the split code twice gives the exact same result)
RANDOM_SEED = 42

# ----------------------------
# CREATE DIRECTORIES IF THEY DON'T EXIST
# ----------------------------
# This runs automatically whenever config.py is imported, so we never
# get a "folder not found" error when saving outputs later.
for directory in [PROCESSED_DATA_DIR, SPLITS_DIR, SAVED_MODELS_DIR,
                   PLOTS_DIR, GRADCAM_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)