"""
Feature extraction utilities for computer vision project.
"""

import cv2
import numpy as np
from skimage.feature import hog
from config import HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM


def extract_hog_features(image):
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: HOG feature vector
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Extract HOG features
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True
    )
    
    return features


def extract_hog_features_batch(images):
    """
    Extract HOG features from a batch of images.
    
    Args:
        images (numpy.ndarray): Batch of images
        
    Returns:
        numpy.ndarray: Array of HOG feature vectors
    """
    features_list = []
    
    for image in images:
        hog_features = extract_hog_features(image)
        features_list.append(hog_features)
    
    return np.array(features_list)


def visualize_hog(image, hog_image_resized=True):
    """
    Visualize HOG features on an image.
    
    Args:
        image (numpy.ndarray): Input image
        hog_image_resized (bool): Whether to resize HOG visualization
        
    Returns:
        numpy.ndarray: Visualization of HOG features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Extract HOG features and visualization
    features, hog_image = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=True,
        feature_vector=True
    )
    
    if hog_image_resized:
        # Resize HOG image to match original image size
        hog_image = cv2.resize(hog_image, (gray.shape[1], gray.shape[0]))
    
    return hog_image


def extract_color_histogram(image, bins=256):
    """
    Extract color histogram features from an image.
    
    Args:
        image (numpy.ndarray): Input image
        bins (int): Number of bins for histogram
        
    Returns:
        numpy.ndarray: Concatenated color histogram features
    """
    histograms = []
    
    # Extract histogram for each channel
    for i in range(image.shape[2]):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    
    return np.concatenate(histograms)


def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract LBP (Local Binary Pattern) features from an image.
    
    Args:
        image (numpy.ndarray): Input image
        radius (int): Radius for LBP
        n_points (int): Number of points for LBP
        
    Returns:
        numpy.ndarray: LBP histogram features
    """
    from skimage.feature import local_binary_pattern
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist
