"""
Evaluation utilities for computer vision project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from config import FIGURES_DIR, METRICS_DIR
from train import load_model
from features import extract_hog_features_batch
from dataset import create_dataset


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        average (str): Averaging method for multi-class metrics
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average)
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        label_encoder: Fitted label encoder
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_classification_metrics(metrics, save_path=None):
    """
    Plot classification metrics as a bar chart.
    
    Args:
        metrics (dict): Dictionary of metrics
        save_path (str): Path to save the plot
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title('Classification Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
    
    plt.show()


def plot_roc_curve_multiclass(y_true, y_proba, label_encoder, save_path=None):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_proba (numpy.ndarray): Predicted probabilities
        label_encoder: Fitted label encoder
        save_path (str): Path to save the plot
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    # Binarize the labels
    n_classes = len(label_encoder.classes_)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def generate_evaluation_report(model, X_test, y_test, label_encoder, save_report=True):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        model: Trained classifier
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        label_encoder: Fitted label encoder
        save_report (bool): Whether to save the report
        
    Returns:
        dict: Complete evaluation results
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, 
                                       target_names=label_encoder.classes_,
                                       output_dict=True)
    
    # Create results dictionary
    results = {
        'metrics': metrics,
        'classification_report': class_report,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    # Save report if requested
    if save_report:
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        # Save metrics as JSON
        import json
        metrics_path = os.path.join(METRICS_DIR, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save plots
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        plot_confusion_matrix(y_test, y_pred, label_encoder,
                            save_path=os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
        
        plot_classification_metrics(metrics,
                                  save_path=os.path.join(FIGURES_DIR, 'classification_metrics.png'))
        
        if y_proba is not None and len(label_encoder.classes_) == 2:
            plot_roc_curve_multiclass(y_test, y_proba, label_encoder,
                                    save_path=os.path.join(FIGURES_DIR, 'roc_curve.png'))
        
        print(f"Evaluation report saved to: {metrics_path}")
    
    return results


def evaluate_saved_model():
    """
    Evaluate a saved model on test data.
    
    Returns:
        dict: Evaluation results
    """
    # Load model and label encoder
    model, label_encoder = load_model()
    
    # Create test dataset (without splitting)
    X_train, X_test, y_train, y_test, _ = create_dataset()
    
    # Extract HOG features
    X_test_hog = extract_hog_features_batch(X_test)
    
    # Generate evaluation report
    results = generate_evaluation_report(model, X_test_hog, y_test, label_encoder)
    
    return results


def compare_models(model_results_dict, save_comparison=True):
    """
    Compare multiple models and create comparison plots.
    
    Args:
        model_results_dict (dict): Dictionary with model names as keys and results as values
        save_comparison (bool): Whether to save the comparison plot
        
    Returns:
        None
    """
    model_names = list(model_results_dict.keys())
    accuracies = [results['metrics']['accuracy'] for results in model_results_dict.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title('Model Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    if save_comparison:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()
