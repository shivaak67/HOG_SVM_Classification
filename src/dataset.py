"""
Dataset module for HOG-SVM Classification project.
Handles image loading, preprocessing, and HOG feature extraction.
"""

import os  
import cv2  
import numpy as np  
from skimage.feature import hog  # Import HOG feature extraction function
from sklearn.model_selection import train_test_split  # Import train/test split function
from config import (
    TRAIN_DIR, TEST_DIR, IMAGE_SIZE, HOG_CONFIG, CLASS_LABELS  # Import configuration values
)


def load_image(image_path):
    """
    Load an image from the given path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format from OpenCV)
    """
    image = cv2.imread(image_path)  # Read image from file path using OpenCV
    if image is None:  # Check if image loading failed
        raise ValueError(f"Failed to load image: {image_path}")  # Raise error if image not found
    return image  # Return loaded image


def preprocess_image(image):
    """
    Preprocess image for HOG feature extraction.
    - Resize to consistent dimensions
    - Convert BGR to RGB
    - Convert to grayscale
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Grayscale image resized to IMAGE_SIZE
    """
    # Resize image
    resized = cv2.resize(image, IMAGE_SIZE)  # Resize image to consistent dimensions from config
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to standard RGB
    
    # Convert to grayscale for HOG
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale for HOG extraction
    
    return gray  # Return preprocessed grayscale image


def extract_hog_features(image):
    """
    Extract HOG features from a grayscale image.
    
    Args:
        image: Grayscale image
        
    Returns:
        HOG feature vector
    """
    features = hog(
        image,  # Input grayscale image
        orientations=HOG_CONFIG["orientations"],  # Number of orientation bins from config
        pixels_per_cell=HOG_CONFIG["pixels_per_cell"],  # Cell size in pixels from config
        cells_per_block=HOG_CONFIG["cells_per_block"],  # Block size in cells from config
        visualize=HOG_CONFIG["visualize"],  # Whether to return visualization image from config
        block_norm=HOG_CONFIG["block_norm"],  # Block normalization method from config
        transform_sqrt=HOG_CONFIG["transform_sqrt"]  # Square root transform from config
    )
    return features  # Return extracted HOG feature vector


def load_class_data(class_dir, label):
    """
    Load all images from a class directory and extract HOG features.
    
    Args:
        class_dir: Path to the class directory
        label: Integer label for this class (0 or 1)
        
    Returns:
        features: List of HOG feature vectors
        labels: List of corresponding labels
    """
    features = []  # Initialize empty list to store HOG feature vectors
    labels = []  # Initialize empty list to store corresponding labels
    
    if not os.path.exists(class_dir):  # Check if directory exists
        print(f"Warning: Directory not found: {class_dir}")  # Print warning if not found
        return features, labels  # Return empty lists if directory doesn't exist
    
    image_files = [f for f in os.listdir(class_dir)  # List all files in directory
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]  # Filter for image files only
    
    for image_file in image_files:  # Iterate through each image file
        image_path = os.path.join(class_dir, image_file)  # Construct full path to image file
        
        try:  # Start error handling block
            # Load and preprocess image
            image = load_image(image_path)  # Load image from path
            gray = preprocess_image(image)  # Preprocess image to grayscale
            
            # Extract HOG features
            hog_features = extract_hog_features(gray)  # Extract HOG features from grayscale image
            
            features.append(hog_features)  # Append HOG features to list
            labels.append(label)  # Append corresponding label to list
            
        except Exception as e:  # Catch any errors during processing
            print(f"Error processing {image_file}: {e}")  # Print error message and continue
    
    return features, labels  # Return lists of features and labels


def load_dataset():
    """
    Load the complete dataset from the training directory.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
    """
    all_features = []  # Initialize list to store all features from all classes
    all_labels = []  # Initialize list to store all labels from all classes
    
    # Load data from each class
    for class_idx, class_name in enumerate(CLASS_LABELS):  # Iterate through class labels with index
        class_dir = os.path.join(TRAIN_DIR, class_name)  # Construct path to class directory
        print(f"Loading {class_name} from {class_dir}")  # Print progress message
        
        features, labels = load_class_data(class_dir, class_idx)  # Load features and labels for this class
        all_features.extend(features)  # Extend list with this class's features
        all_labels.extend(labels)  # Extend list with this class's labels
        
        print(f"  Loaded {len(features)} images from {class_name}")  # Print count of loaded images
    
    # Convert to numpy arrays
    X = np.array(all_features)  # Convert features list to numpy array
    y = np.array(all_labels)  # Convert labels list to numpy array
    
    print(f"\nTotal dataset: {X.shape[0]} samples, {X.shape[1]} features")  # Print dataset summary
    
    return X, y  # Return feature matrix and label array


def load_test_dataset():
    """
    Load the test dataset from the test directory.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
    """
    all_features = []  # Initialize list to store all test features
    all_labels = []  # Initialize list to store all test labels
    
    # Load data from each class
    for class_idx, class_name in enumerate(CLASS_LABELS):  # Iterate through class labels with index
        class_dir = os.path.join(TEST_DIR, class_name)  # Construct path to test class directory
        print(f"Loading {class_name} from {class_dir}")  # Print progress message
        
        features, labels = load_class_data(class_dir, class_idx)  # Load features and labels for this class
        all_features.extend(features)  # Extend list with this class's features
        all_labels.extend(labels)  # Extend list with this class's labels
        
        print(f"  Loaded {len(features)} images from {class_name}")  # Print count of loaded images
    
    # Convert to numpy arrays
    X = np.array(all_features)  # Convert features list to numpy array
    y = np.array(all_labels)  # Convert labels list to numpy array
    
    print(f"\nTest dataset: {X.shape[0]} samples, {X.shape[1]} features")  # Print test dataset summary
    
    return X, y  # Return test feature matrix and label array


def get_train_test_split(test_size=0.2, random_state=42):
    """
    Split the training data into train and validation sets.
    
    Args:
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    X, y = load_dataset()  # Load complete dataset
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y  # Split with stratification to maintain class balance
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")  # Print training set size
    print(f"Validation set: {X_val.shape[0]} samples")  # Print validation set size
    
    return X_train, X_val, y_train, y_val  # Return split datasets
