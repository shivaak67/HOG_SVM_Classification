"""
Training module for HOG-SVM Classification project.
Handles SVM model training, feature scaling, and model persistence.
"""

import os  # Import operating system interface for file operations
import pickle  # Import pickle for saving/loading Python objects
import numpy as np  # Import NumPy for array operations
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature normalization
from sklearn.metrics import accuracy_score  # Import accuracy metric for evaluation
from config import (  # Import configuration values
    SVM_CONFIG, MODEL_PATH, SCALER_PATH, MODELS_DIR
)
from dataset import get_train_test_split  # Import dataset loading function


def create_model_directory():
    """
    Create the models directory if it doesn't exist.
    """
    if not os.path.exists(MODELS_DIR):  # Check if models directory exists
        os.makedirs(MODELS_DIR)  # Create directory if it doesn't exist
        print(f"Created directory: {MODELS_DIR}")  # Print directory creation message


def train_model():
    """
    Train the SVM classifier on the dataset.
    
    Returns:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        accuracy: Validation accuracy
    """
    # Load training and validation data
    print("Loading dataset...")  # Print progress message
    X_train, X_val, y_train, y_val = get_train_test_split()  # Get train/validation split
    
    # Scale features
    print("\nScaling features...")  # Print progress message
    scaler = StandardScaler()  # Initialize StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data and transform
    X_val_scaled = scaler.transform(X_val)  # Transform validation data using fitted scaler
    
    # Initialize SVM model
    print("Initializing SVM model...")  # Print progress message
    model = SVC(**SVM_CONFIG)  # Create SVM with parameters from config
    
    # Train model
    print("Training model...")  # Print progress message
    model.fit(X_train_scaled, y_train)  # Fit SVM on scaled training data
    
    # Validate model
    print("Validating model...")  # Print progress message
    y_pred = model.predict(X_val_scaled)  # Make predictions on validation set
    accuracy = accuracy_score(y_val, y_pred)  # Calculate validation accuracy
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")  # Print validation accuracy
    
    return model, scaler, accuracy  # Return trained model, scaler, and accuracy


def save_model(model, scaler):
    """
    Save the trained model and scaler to disk.
    
    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
    """
    create_model_directory()  # Ensure models directory exists
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")  # Print save message
    with open(MODEL_PATH, 'wb') as f:  # Open file in binary write mode
        pickle.dump(model, f)  # Serialize and save model to file
    print("Model saved successfully.")  # Print success message
    
    # Save scaler
    print(f"Saving scaler to {SCALER_PATH}...")  # Print save message
    with open(SCALER_PATH, 'wb') as f:  # Open file in binary write mode
        pickle.dump(scaler, f)  # Serialize and save scaler to file
    print("Scaler saved successfully.")  # Print success message


def load_model():
    """
    Load the trained model and scaler from disk.
    
    Returns:
        model: Loaded SVM model
        scaler: Loaded StandardScaler
    """
    # Load model
    print(f"Loading model from {MODEL_PATH}...")  # Print load message
    with open(MODEL_PATH, 'rb') as f:  # Open file in binary read mode
        model = pickle.load(f)  # Deserialize and load model from file
    print("Model loaded successfully.")  # Print success message
    
    # Load scaler
    print(f"Loading scaler from {SCALER_PATH}...")  # Print load message
    with open(SCALER_PATH, 'rb') as f:  # Open file in binary read mode
        scaler = pickle.load(f)  # Deserialize and load scaler from file
    print("Scaler loaded successfully.")  # Print success message
    
    return model, scaler  # Return loaded model and scaler


if __name__ == "__main__":  # Check if script is run directly
    print("=" * 50)  # Print separator line
    print("HOG-SVM Training Pipeline")  # Print title
    print("=" * 50)  # Print separator line
    
    # Train model
    model, scaler, accuracy = train_model()  # Execute training pipeline
    
    # Save model
    save_model(model, scaler)  # Persist model and scaler to disk
    
    print("\n" + "=" * 50)  # Print separator line
    print("Training completed successfully!")  # Print completion message
    print("=" * 50)  # Print separator line
