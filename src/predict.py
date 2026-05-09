"""
Prediction module for HOG-SVM Classification project.
Handles inference on new images using the trained model.
"""

import os  # Import operating system interface for file operations
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for array operations
from config import (  # Import configuration values
    IMAGE_SIZE, HOG_CONFIG, CLASS_LABELS
)
from train import load_model  # Import model loading function


def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        gray: Preprocessed grayscale image
    """
    # Load image
    image = cv2.imread(image_path)  # Read image from file path
    if image is None:  # Check if image loading failed
        raise ValueError(f"Failed to load image: {image_path}")  # Raise error if not found
    
    # Resize image
    resized = cv2.resize(image, IMAGE_SIZE)  # Resize to consistent dimensions
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to RGB
    
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale for HOG
    
    return gray  # Return preprocessed grayscale image


def extract_hog_features_single(image):
    """
    Extract HOG features from a single preprocessed image.
    
    Args:
        image: Preprocessed grayscale image
        
    Returns:
        features: HOG feature vector
    """
    from skimage.feature import hog  # Import HOG function
    
    features = hog(  # Extract HOG features
        image,  # Input grayscale image
        orientations=HOG_CONFIG["orientations"],  # Number of orientation bins
        pixels_per_cell=HOG_CONFIG["pixels_per_cell"],  # Cell size in pixels
        cells_per_block=HOG_CONFIG["cells_per_block"],  # Block size in cells
        visualize=HOG_CONFIG["visualize"],  # Whether to return visualization
        block_norm=HOG_CONFIG["block_norm"],  # Block normalization method
        transform_sqrt=HOG_CONFIG["transform_sqrt"]  # Square root transform
    )
    
    return features  # Return feature vector


def predict_single_image(image_path, return_probabilities=False):
    """
    Predict the class of a single image.
    
    Args:
        image_path: Path to the image file
        return_probabilities: Whether to return class probabilities
        
    Returns:
        prediction: Predicted class label
        probabilities: Class probabilities (if requested)
    """
    # Load trained model and scaler
    model, scaler = load_model()  # Load model and scaler from disk
    
    # Preprocess image
    gray = preprocess_single_image(image_path)  # Preprocess the image
    
    # Extract HOG features
    features = extract_hog_features_single(gray)  # Extract features from image
    
    # Reshape features for prediction (1 sample)
    features_reshaped = features.reshape(1, -1)  # Reshape to (1, n_features)
    
    # Scale features
    features_scaled = scaler.transform(features_reshaped)  # Apply fitted scaler
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]  # Predict class label
    
    # Get probabilities if requested
    probabilities = None  # Initialize probabilities
    if return_probabilities:  # Check if probabilities requested
        probabilities = model.predict_proba(features_scaled)[0]  # Get class probabilities
    
    # Convert label to class name
    class_name = CLASS_LABELS[prediction]  # Map label to class name
    
    if return_probabilities:  # Return based on request
        return class_name, probabilities  # Return prediction and probabilities
    else:
        return class_name  # Return just prediction


def predict_batch_images(image_paths, return_probabilities=False):
    """
    Predict classes for multiple images.
    
    Args:
        image_paths: List of image file paths
        return_probabilities: Whether to return class probabilities
        
    Returns:
        predictions: List of predicted class names
        all_probabilities: List of class probabilities (if requested)
    """
    # Load trained model and scaler
    model, scaler = load_model()  # Load model and scaler from disk
    
    predictions = []  # Initialize predictions list
    all_probabilities = []  # Initialize probabilities list
    
    for image_path in image_paths:  # Iterate through image paths
        try:  # Start error handling
            # Preprocess image
            gray = preprocess_single_image(image_path)  # Preprocess the image
            
            # Extract HOG features
            features = extract_hog_features_single(gray)  # Extract features
            
            # Reshape features
            features_reshaped = features.reshape(1, -1)  # Reshape for prediction
            
            # Scale features
            features_scaled = scaler.transform(features_reshaped)  # Apply scaler
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]  # Predict class
            class_name = CLASS_LABELS[prediction]  # Map to class name
            predictions.append(class_name)  # Add to predictions list
            
            # Get probabilities if requested
            if return_probabilities:  # Check if probabilities requested
                probabilities = model.predict_proba(features_scaled)[0]  # Get probabilities
                all_probabilities.append(probabilities)  # Add to list
                
        except Exception as e:  # Catch any errors
            print(f"Error processing {image_path}: {e}")  # Print error message
            predictions.append("Error")  # Add error placeholder
            if return_probabilities:  # Handle probabilities list
                all_probabilities.append([0.0, 0.0])  # Add placeholder probabilities
    
    if return_probabilities:  # Return based on request
        return predictions, all_probabilities  # Return predictions and probabilities
    else:
        return predictions  # Return just predictions


def print_prediction_result(image_path, prediction, probabilities=None):
    """
    Print prediction results in a formatted way.
    
    Args:
        image_path: Path to the image file
        prediction: Predicted class name
        probabilities: Class probabilities (optional)
    """
    print(f"\nImage: {os.path.basename(image_path)}")  # Print image filename
    print(f"Prediction: {prediction}")  # Print predicted class
    
    if probabilities is not None:  # Check if probabilities provided
        print("Probabilities:")  # Print probabilities header
        for i, class_name in enumerate(CLASS_LABELS):  # Iterate through classes
            prob = probabilities[i]  # Get probability for this class
            print(f"  {class_name}: {prob:.4f}")  # Print probability with 4 decimal places


if __name__ == "__main__":  # Check if script is run directly
    # Example usage
    print("=" * 50)  # Print separator line
    print("HOG-SVM Prediction Demo")  # Print title
    print("=" * 50)  # Print separator line
    
    # Example image path (replace with actual path)
    example_path = "../data/raw/train/class_0/your_image.jpg"  # Example path
    
    if os.path.exists(example_path):  # Check if example image exists
        print(f"Predicting for: {example_path}")  # Print image path
        
        # Make prediction with probabilities
        prediction, probs = predict_single_image(example_path, return_probabilities=True)  # Get prediction
        
        # Print results
        print_prediction_result(example_path, prediction, probs)  # Display results
    else:  # If example doesn't exist
        print(f"Example image not found: {example_path}")  # Print error
        print("Replace with actual image path to test predictions")  # Print instruction
    
    print("=" * 50)  # Print separator line
    print("Prediction demo completed!")  # Print completion message
    print("=" * 50)  # Print separator line