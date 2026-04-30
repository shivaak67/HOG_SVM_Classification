"""
Evaluation module for HOG-SVM Classification project.
Handles model evaluation metrics, confusion matrix, and performance reporting.
"""

import os  # Import operating system interface for file operations
import numpy as np  # Import NumPy for array operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.metrics import (  # Import evaluation metrics from sklearn
    accuracy_score, precision_score, recall_score, f1_score,  # Basic metrics
    confusion_matrix, classification_report  # Advanced metrics
)
import seaborn as sns  # Import seaborn for enhanced visualization
from config import CLASS_LABELS  # Import class labels from config
from dataset import load_test_dataset  # Import test dataset loading function
from train import load_model  # Import model loading function


def evaluate_model():
    """
    Evaluate the trained SVM model on the test dataset.
    
    Returns:
        metrics: Dictionary containing evaluation metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    # Load test dataset
    print("Loading test dataset...")  # Print progress message
    X_test, y_true = load_test_dataset()  # Load test features and labels
    
    # Load trained model and scaler
    print("Loading trained model...")  # Print progress message
    model, scaler = load_model()  # Load model and scaler from disk
    
    # Scale test features
    print("Scaling test features...")  # Print progress message
    X_test_scaled = scaler.transform(X_test)  # Transform test data using fitted scaler
    
    # Make predictions
    print("Making predictions...")  # Print progress message
    y_pred = model.predict(X_test_scaled)  # Predict labels for test set
    
    # Calculate metrics
    print("Calculating metrics...")  # Print progress message
    accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy
    precision = precision_score(y_true, y_pred, average='weighted')  # Calculate weighted precision
    recall = recall_score(y_true, y_pred, average='weighted')  # Calculate weighted recall
    f1 = f1_score(y_true, y_pred, average='weighted')  # Calculate weighted F1 score
    
    # Store metrics in dictionary
    metrics = {  # Create metrics dictionary
        'accuracy': accuracy,  # Store accuracy
        'precision': precision,  # Store precision
        'recall': recall,  # Store recall
        'f1_score': f1  # Store F1 score
    }
    
    return metrics, y_true, y_pred  # Return metrics and predictions


def print_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 50)  # Print separator line
    print("Evaluation Metrics")  # Print title
    print("=" * 50)  # Print separator line
    print(f"Accuracy:  {metrics['accuracy']:.4f}")  # Print accuracy
    print(f"Precision: {metrics['precision']:.4f}")  # Print precision
    print(f"Recall:    {metrics['recall']:.4f}")  # Print recall
    print(f"F1 Score:  {metrics['f1_score']:.4f}")  # Print F1 score
    print("=" * 50)  # Print separator line


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and optionally save the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
    
    # Create figure
    plt.figure(figsize=(8, 6))  # Create figure with specified size
    
    # Plot heatmap
    sns.heatmap(  # Create heatmap visualization
        cm,  # Confusion matrix data
        annot=True,  # Annotate cells with values
        fmt='d',  # Use integer format
        cmap='Blues',  # Use blue color map
        xticklabels=CLASS_LABELS,  # Set x-axis labels
        yticklabels=CLASS_LABELS  # Set y-axis labels
    )
    
    # Add labels and title
    plt.xlabel('Predicted Label')  # Set x-axis label
    plt.ylabel('True Label')  # Set y-axis label
    plt.title('Confusion Matrix')  # Set plot title
    
    # Save or show plot
    if save_path:  # Check if save path is provided
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure to file
        print(f"Confusion matrix saved to {save_path}")  # Print save confirmation
    else:  # If no save path
        plt.show()  # Display plot interactively
    
    plt.close()  # Close figure to free memory


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "=" * 50)  # Print separator line
    print("Classification Report")  # Print title
    print("=" * 50)  # Print separator line
    print(classification_report(  # Print classification report
        y_true,  # True labels
        y_pred,  # Predicted labels
        target_names=CLASS_LABELS  # Class names for report
    ))
    print("=" * 50)  # Print separator line


if __name__ == "__main__":  # Check if script is run directly
    print("=" * 50)  # Print separator line
    print("HOG-SVM Evaluation Pipeline")  # Print title
    print("=" * 50)  # Print separator line
    
    # Evaluate model
    metrics, y_true, y_pred = evaluate_model()  # Run evaluation
    
    # Print metrics
    print_metrics(metrics)  # Display evaluation metrics
    
    # Print classification report
    print_classification_report(y_true, y_pred)  # Display detailed report
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)  # Display confusion matrix
    
    print("\n" + "=" * 50)  # Print separator line
    print("Evaluation completed successfully!")  # Print completion message
    print("=" * 50)  # Print separator line
