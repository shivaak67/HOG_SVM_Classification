"""
Configuration file for computer vision project.
"""

import os

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")

# Model paths
MODEL_DIR = "models"
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Output paths
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

# SVM parameters
SVM_C = 1.0
SVM_KERNEL = 'rbf'
SVM_GAMMA = 'scale'

# Image parameters
IMAGE_SIZE = (64, 64)
COLOR_SPACE = 'RGB'
