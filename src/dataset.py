"""
Dataset utilities for computer vision project.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import TRAIN_DIR, TEST_DIR
from preprocess import load_images_from_directory, preprocess_images, create_label_encoder


def create_dataset(test_size=0.2, random_state=42):
    """
    Create training and testing datasets from raw images.
    
    Args:
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    # Load training data
    print("Loading training data...")
    train_images, train_labels = load_images_from_directory(TRAIN_DIR)
    
    # Load test data if available, otherwise split training data
    if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR):
        print("Loading test data...")
        test_images, test_labels = load_images_from_directory(TEST_DIR)
    else:
        print("No test directory found, splitting training data...")
        # Combine all data and split
        all_images = train_images
        all_labels = train_labels
        
        # Split data
        train_images, test_images, train_labels, test_labels = train_test_split(
            all_images, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
        )
    
    # Preprocess images
    print("Preprocessing images...")
    X_train = preprocess_images(train_images)
    X_test = preprocess_images(test_images)
    
    # Create label encoder
    print("Creating label encoder...")
    label_encoder, y_train = create_label_encoder(train_labels)
    _, y_test = create_label_encoder(test_labels)
    
    print(f"Dataset created:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Classes: {label_encoder.classes_}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def get_class_distribution(labels, label_encoder=None):
    """
    Get the distribution of classes in the dataset.
    
    Args:
        labels (array-like): Array of labels
        label_encoder (LabelEncoder): Optional label encoder to get class names
        
    Returns:
        dict: Dictionary with class names as keys and counts as values
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if label_encoder is not None:
        class_names = label_encoder.inverse_transform(unique_labels)
    else:
        class_names = unique_labels
    
    distribution = dict(zip(class_names, counts))
    
    return distribution


def balance_dataset(X, y, method='oversample'):
    """
    Balance the dataset using oversampling or undersampling.
    
    Args:
        X (numpy.ndarray): Feature array
        y (numpy.ndarray): Label array
        method (str): 'oversample' or 'undersample'
        
    Returns:
        tuple: (balanced_X, balanced_y)
    """
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Method must be 'oversample' or 'undersample'")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Create data loaders for training and testing.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader) - Note: This is a placeholder
                for deep learning frameworks. For scikit-learn, we just
                return the data as is.
    """
    # For scikit-learn, we don't need data loaders
    # This function is provided for compatibility with deep learning frameworks
    
    print(f"Data ready for training:")
    print(f"  Training batch size: {batch_size}")
    print(f"  Number of training batches: {len(X_train) // batch_size}")
    print(f"  Number of test batches: {len(X_test) // batch_size}")
    
    return X_train, y_train, X_test, y_test
