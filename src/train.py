"""
Training utilities for computer vision project.
"""

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from config import SVM_MODEL_PATH, LABEL_ENCODER_PATH, SVM_C, SVM_KERNEL, SVM_GAMMA
from features import extract_hog_features_batch
from dataset import create_dataset


def train_svm_classifier(X_train, y_train, param_grid=None, cv=5):
    """
    Train an SVM classifier with optional hyperparameter tuning.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        param_grid (dict): Parameter grid for GridSearchCV
        cv (int): Number of cross-validation folds
        
    Returns:
        tuple: (trained_model, best_params)
    """
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    
    # Create SVM classifier
    svm = SVC(random_state=42)
    
    # Perform grid search
    print("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_model, best_params


def train_simple_svm(X_train, y_train):
    """
    Train a simple SVM classifier with default parameters.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        
    Returns:
        sklearn.svm.SVC: Trained SVM classifier
    """
    print("Training SVM classifier...")
    svm = SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA, random_state=42)
    svm.fit(X_train, y_train)
    
    return svm


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained classifier
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        label_encoder: Label encoder for class names
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    
    # Generate classification report
    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        target_names = None
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def save_model(model, label_encoder, model_path=SVM_MODEL_PATH, encoder_path=LABEL_ENCODER_PATH):
    """
    Save the trained model and label encoder.
    
    Args:
        model: Trained classifier
        label_encoder: Fitted label encoder
        model_path (str): Path to save the model
        encoder_path (str): Path to save the label encoder
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")


def load_model(model_path=SVM_MODEL_PATH, encoder_path=LABEL_ENCODER_PATH):
    """
    Load a trained model and label encoder.
    
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
    
    print(f"Model loaded from: {model_path}")
    print(f"Label encoder loaded from: {encoder_path}")
    
    return model, label_encoder


def perform_cross_validation(model, X, y, cv=5):
    """
    Perform cross-validation on the model.
    
    Args:
        model: Classifier to evaluate
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Cross-validation results
    """
    print("Performing cross-validation...")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }


def train_and_evaluate_pipeline(use_grid_search=True, balance_data=False):
    """
    Complete pipeline for training and evaluating the model.
    
    Args:
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        balance_data (bool): Whether to balance the dataset
        
    Returns:
        tuple: (model, label_encoder, evaluation_results)
    """
    # Create dataset
    X_train, X_test, y_train, y_test, label_encoder = create_dataset()
    
    # Balance data if requested
    if balance_data:
        print("Balancing training data...")
        from dataset import balance_dataset
        X_train, y_train = balance_dataset(X_train, y_train, method='oversample')
    
    # Extract HOG features
    print("Extracting HOG features...")
    X_train_hog = extract_hog_features_batch(X_train)
    X_test_hog = extract_hog_features_batch(X_test)
    
    # Train model
    if use_grid_search:
        model, best_params = train_svm_classifier(X_train_hog, y_train)
    else:
        model = train_simple_svm(X_train_hog, y_train)
        best_params = None
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test_hog, y_test, label_encoder)
    
    # Save model
    save_model(model, label_encoder)
    
    return model, label_encoder, evaluation_results
