"""
Prediction utilities for computer vision project.
"""

import os
import cv2
import numpy as np
import pickle
from config import SVM_MODEL_PATH, LABEL_ENCODER_PATH, IMAGE_SIZE, COLOR_SPACE
from features import extract_hog_features


def load_trained_model(model_path=SVM_MODEL_PATH, encoder_path=LABEL_ENCODER_PATH):
    """
    Load a trained model and label encoder for prediction.
    
    Args:
        model_path (str): Path to the saved model
        encoder_path (str): Path to the saved label encoder
        
    Returns:
        tuple: (model, label_encoder)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder


def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize image
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Convert color space if needed
    if COLOR_SPACE == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    return img


def predict_single_image(image_path, model=None, label_encoder=None, return_probabilities=False):
    """
    Make prediction on a single image.
    
    Args:
        image_path (str): Path to the image file
        model: Trained model (if None, will load from file)
        label_encoder: Label encoder (if None, will load from file)
        return_probabilities (bool): Whether to return prediction probabilities
        
    Returns:
        dict: Prediction results
    """
    # Load model and encoder if not provided
    if model is None or label_encoder is None:
        model, label_encoder = load_trained_model()
    
    # Preprocess image
    img = preprocess_single_image(image_path)
    
    # Extract HOG features
    features = extract_hog_features(img)
    
    # Make prediction
    prediction = model.predict(features.reshape(1, -1))[0]
    
    # Get class name
    class_name = label_encoder.inverse_transform([prediction])[0]
    
    results = {
        'prediction': prediction,
        'class_name': class_name,
        'image_path': image_path
    }
    
    # Add probabilities if requested and available
    if return_probabilities and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        
        # Create probability dictionary with class names
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            class_name_i = label_encoder.inverse_transform([i])[0]
            prob_dict[class_name_i] = float(prob)
        
        results['probabilities'] = prob_dict
        results['confidence'] = float(max(probabilities))
    
    return results


def predict_batch_images(image_paths, model=None, label_encoder=None, return_probabilities=False):
    """
    Make predictions on a batch of images.
    
    Args:
        image_paths (list): List of image file paths
        model: Trained model (if None, will load from file)
        label_encoder: Label encoder (if None, will load from file)
        return_probabilities (bool): Whether to return prediction probabilities
        
    Returns:
        list: List of prediction results
    """
    # Load model and encoder if not provided
    if model is None or label_encoder is None:
        model, label_encoder = load_trained_model()
    
    results = []
    
    for image_path in image_paths:
        try:
            result = predict_single_image(
                image_path, model, label_encoder, return_probabilities
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def predict_directory(directory_path, model=None, label_encoder=None, return_probabilities=False, 
                      save_predictions=True):
    """
    Make predictions on all images in a directory.
    
    Args:
        directory_path (str): Path to directory containing images
        model: Trained model (if None, will load from file)
        label_encoder: Label encoder (if None, will load from file)
        return_probabilities (bool): Whether to return prediction probabilities
        save_predictions (bool): Whether to save predictions to file
        
    Returns:
        list: List of prediction results
    """
    # Get all image files in directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(directory_path, filename))
    
    if not image_paths:
        raise ValueError(f"No images found in {directory_path}")
    
    print(f"Found {len(image_paths)} images in {directory_path}")
    
    # Make predictions
    results = predict_batch_images(image_paths, model, label_encoder, return_probabilities)
    
    # Save predictions if requested
    if save_predictions:
        from config import PREDICTIONS_DIR
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        
        import json
        save_path = os.path.join(PREDICTIONS_DIR, 'batch_predictions.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions saved to: {save_path}")
    
    return results


def visualize_prediction(image_path, prediction_result, save_path=None):
    """
    Visualize an image with its prediction.
    
    Args:
        image_path (str): Path to the image file
        prediction_result (dict): Prediction result dictionary
        save_path (str): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Load and display image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')
    
    # Add prediction text
    class_name = prediction_result.get('class_name', 'Unknown')
    confidence = prediction_result.get('confidence', 0)
    
    title = f"Prediction: {class_name}"
    if confidence > 0:
        title += f" (Confidence: {confidence:.3f})"
    
    plt.title(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def create_prediction_summary(prediction_results):
    """
    Create a summary of prediction results.
    
    Args:
        prediction_results (list): List of prediction results
        
    Returns:
        dict: Summary statistics
    """
    total_images = len(prediction_results)
    successful_predictions = sum(1 for r in prediction_results if 'error' not in r)
    failed_predictions = total_images - successful_predictions
    
    if successful_predictions == 0:
        return {
            'total_images': total_images,
            'successful_predictions': 0,
            'failed_predictions': failed_predictions,
            'class_distribution': {}
        }
    
    # Count predictions by class
    class_counts = {}
    for result in prediction_results:
        if 'error' not in result:
            class_name = result.get('class_name', 'Unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Calculate percentages
    class_percentages = {}
    for class_name, count in class_counts.items():
        class_percentages[class_name] = (count / successful_predictions) * 100
    
    summary = {
        'total_images': total_images,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'class_distribution': class_counts,
        'class_percentages': class_percentages
    }
    
    return summary


def interactive_prediction():
    """
    Interactive function to make predictions on user-provided images.
    """
    print("Interactive Image Prediction")
    print("=" * 40)
    
    # Load model and encoder
    model, label_encoder = load_trained_model()
    
    print(f"Model loaded. Classes: {list(label_encoder.classes_)}")
    print("\nEnter image path (or 'quit' to exit):")
    
    while True:
        image_path = input("> ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        try:
            # Make prediction
            result = predict_single_image(image_path, model, label_encoder, return_probabilities=True)
            
            # Display result
            print(f"\nPrediction: {result['class_name']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.3f}")
            
            if 'probabilities' in result:
                print("Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name}: {prob:.3f}")
            
            # Ask if user wants to visualize
            visualize_choice = input("\nVisualize image? (y/n): ").strip().lower()
            if visualize_choice == 'y':
                visualize_prediction(image_path, result)
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\nEnter next image path (or 'quit' to exit):")
