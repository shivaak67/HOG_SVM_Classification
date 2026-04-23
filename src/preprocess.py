"""
Data preprocessing utilities for computer vision project.
"""

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import IMAGE_SIZE, COLOR_SPACE, PROCESSED_DATA_DIR


def load_images_from_directory(directory):
    """
    Load images from a directory and return images and labels.
    
    Args:
        directory (str): Path to directory containing class subdirectories
        
    Returns:
        tuple: (images, labels) where images is a list of numpy arrays
               and labels is a list of strings
    """
    images = []
    labels = []
    
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, IMAGE_SIZE)
                    
                    # Convert color space if needed
                    if COLOR_SPACE == 'RGB':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    images.append(img)
                    labels.append(class_name)
    
    return images, labels


def preprocess_images(images):
    """
    Preprocess a list of images.
    
    Args:
        images (list): List of numpy arrays representing images
        
    Returns:
        numpy.ndarray: Preprocessed images array
    """
    processed_images = []
    
    for img in images:
        # Normalize pixel values to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        processed_images.append(img_normalized)
    
    return np.array(processed_images)


def create_label_encoder(labels):
    """
    Create and fit a label encoder.
    
    Args:
        labels (list): List of string labels
        
    Returns:
        tuple: (label_encoder, encoded_labels)
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return label_encoder, encoded_labels


def save_preprocessed_data(images, labels, filename_prefix):
    """
    Save preprocessed data to the processed directory.
    
    Args:
        images (numpy.ndarray): Preprocessed images
        labels (numpy.ndarray): Encoded labels
        filename_prefix (str): Prefix for saved files (e.g., 'train', 'test')
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    images_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_images.npy")
    labels_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_labels.npy")
    
    np.save(images_path, images)
    np.save(labels_path, labels)
    
    print(f"Saved {filename_prefix} data:")
    print(f"  Images: {images_path}")
    print(f"  Labels: {labels_path}")


def load_preprocessed_data(filename_prefix):
    """
    Load preprocessed data from the processed directory.
    
    Args:
        filename_prefix (str): Prefix for saved files (e.g., 'train', 'test')
        
    Returns:
        tuple: (images, labels)
    """
    images_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_images.npy")
    labels_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_labels.npy")
    
    images = np.load(images_path)
    labels = np.load(labels_path)
    
    return images, labels
