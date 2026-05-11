"""
Configuration file for HOG-SVM Classification project.
Contains all paths, hyperparameters, and settings used across the project.
"""

import os

# Get the base directory of the project (parent of src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")           # Main data folder
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")       # Original, unprocessed images
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")  # Processed features/data
MODELS_DIR = os.path.join(BASE_DIR, "models")      # Saved models and scalers

# Training and testing data paths
TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")    # Training images (class_0, class_1)
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")      # Test images for evaluation


# HOG FEATURE EXTRACTION PARAMETERS
HOG_CONFIG = {
    "orientations": 9,          # Number of gradient orientation bins (0-180 degrees)
    "pixels_per_cell": (8, 8),  # Size of each cell in pixels (smaller = more features)
    "cells_per_block": (2, 2),  # Number of cells in each block (normalization context)
    "visualize": False,         # Return HOG image visualization (set True for debugging)
    "block_norm": "L2-Hys",     # Block normalization method (L2-Hys works well)
    "transform_sqrt": True      # Apply square root transform to reduce illumination effects
}


# SVM CLASSIFIER HYPERPARAMETERS
SVM_CONFIG = {
    "C": 1.0,                   # Regularization parameter (higher = less regularization)
    "kernel": "linear",            # Kernel type: 'linear' for linear SVM
    "gamma": "scale",           # Kernel coefficient: 'scale' uses 1/(n_features * X.var())
    "probability": True,       # Enable probability estimates for predictions
    "random_state": 42          # Random seed for reproducibility
}


# MODEL PERSISTENCE PATHS
MODEL_PATH = os.path.join(MODELS_DIR, "hog_svm_model.pkl")  # Trained SVM model
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")       # Feature scaler (StandardScaler)


# IMAGE PREPROCESSING
IMAGE_SIZE = (128, 128)  # Resize all images to this size for consistent HOG features


# CLASS LABELS
CLASS_LABELS = ["class_0", "class_1"]  # Names of the two classes being classified
