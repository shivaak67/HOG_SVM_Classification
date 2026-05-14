"""Model loading and image preprocessing helpers for the FastAPI app."""

from pathlib import Path
import pickle

import cv2
import numpy as np
from skimage.feature import hog


# ---------------------------------------------------------------------------
# Easy-to-edit inference configuration
# Keep these values in sync with the settings used during model training.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "hog_svm_model.pkl"
FALLBACK_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

IMAGE_SIZE = (128, 128)

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}

# Update this mapping if your training labels use the opposite order.
CLASS_NAMES = {
    0: "cat",
    1: "dog",
}


class ModelNotFoundError(FileNotFoundError):
    """Raised when model files are missing from backend/models."""


def resolve_model_path(primary_path: Path, fallback_path: Path):
    """Prefer backend/models, but support the existing project model location."""
    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise ModelNotFoundError(f"Missing model file: {primary_path}")


def load_pickle(path: Path):
    """Load a pickle file from disk."""
    if not path.exists():
        raise ModelNotFoundError(f"Missing model file: {path}")

    with path.open("rb") as file:
        return pickle.load(file)


def load_artifacts():
    """Load the trained SVM model and fitted scaler."""
    model_path = resolve_model_path(MODEL_PATH, FALLBACK_MODEL_PATH)
    scaler_path = resolve_model_path(SCALER_PATH, FALLBACK_SCALER_PATH)
    model = load_pickle(model_path)
    scaler = load_pickle(scaler_path)
    return model, scaler


def decode_image(image_bytes: bytes):
    """Decode uploaded image bytes into an OpenCV BGR image."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read the uploaded file as an image.")

    return image


def preprocess_image(image):
    """Resize and convert image to grayscale exactly like training."""
    resized = cv2.resize(image, IMAGE_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray


def extract_hog_features(gray_image):
    """Extract one HOG feature vector from a grayscale image."""
    features = hog(
        gray_image,
        orientations=HOG_CONFIG["orientations"],
        pixels_per_cell=HOG_CONFIG["pixels_per_cell"],
        cells_per_block=HOG_CONFIG["cells_per_block"],
        visualize=False,
        block_norm=HOG_CONFIG["block_norm"],
        transform_sqrt=HOG_CONFIG["transform_sqrt"],
    )
    return features.reshape(1, -1)


def get_confidence(model, scaled_features, predicted_label):
    """Return a confidence score when the trained model supports it."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        return float(probabilities[int(predicted_label)])

    if hasattr(model, "decision_function"):
        scores = model.decision_function(scaled_features)
        scores = np.atleast_1d(scores).astype(float)

        if scores.size == 1:
            return float(1 / (1 + np.exp(-abs(scores[0]))))

        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()
        return float(probabilities[int(predicted_label)])

    return None


def predict_image(image_bytes: bytes, model, scaler):
    """Run the complete image-to-prediction pipeline."""
    image = decode_image(image_bytes)
    gray = preprocess_image(image)
    features = extract_hog_features(gray)
    scaled_features = scaler.transform(features)

    predicted_label = int(model.predict(scaled_features)[0])
    prediction = CLASS_NAMES.get(predicted_label, str(predicted_label))
    confidence = get_confidence(model, scaled_features, predicted_label)

    return {
        "prediction": prediction,
        "confidence": confidence,
    }
