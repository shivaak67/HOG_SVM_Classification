"""
Model ensemble module for HOG-SVM Classification project.
Trains multiple SVM models with different parameters and combines predictions.
"""

import os  # Import operating system interface for file operations
import pickle  # Import pickle for model serialization
import numpy as np  # Import NumPy for array operations
from sklearn.svm import SVC  # Import SVM classifier
from sklearn.preprocessing import StandardScaler  # Import feature scaler
from sklearn.model_selection import train_test_split  # Import train/test split
from sklearn.metrics import accuracy_score  # Import accuracy metric

from config import SVM_CONFIG, MODELS_DIR, MODEL_PATH, SCALER_PATH  # Import configuration
from dataset import load_dataset  # Import dataset loading function


def train_ensemble_models():
    """
    Train multiple SVM models with different hyperparameters.
    
    Returns:
        models: List of trained models
        scalers: List of trained scalers
        configs: List of model configurations
    """
    print("Training ensemble models...")  # Print progress message
    
    # Load dataset
    X, y = load_dataset()  # Load complete dataset
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  # Split into train/validation sets
    
    # Define ensemble configurations
    ensemble_configs = [
        {"C": 0.1, "kernel": "linear", "name": "linear_c0.1"},
        {"C": 1.0, "kernel": "linear", "name": "linear_c1.0"},
        {"C": 10.0, "kernel": "linear", "name": "linear_c10.0"},
        {"C": 0.1, "kernel": "rbf", "name": "rbf_c0.1"},
        {"C": 1.0, "kernel": "rbf", "name": "rbf_c1.0"},
        {"C": 10.0, "kernel": "rbf", "name": "rbf_c10.0"}
    ]  # Different model configurations
    
    models = []  # Initialize list for models
    scalers = []  # Initialize list for scalers
    configs = []  # Initialize list for configurations
    
    # Train each model
    for config in ensemble_configs:  # Iterate through configurations
        print(f"\nTraining model: {config['name']}")  # Print model name
        
        # Scale features
        scaler = StandardScaler()  # Create scaler
        X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
        X_val_scaled = scaler.transform(X_val)  # Transform validation data
        
        # Initialize and train model
        model = SVC(
            C=config["C"],  # Regularization parameter
            kernel=config["kernel"],  # Kernel type
            gamma="scale",  # Kernel coefficient
            probability=True,  # Enable probabilities
            random_state=42  # Random seed
        )  # Create SVM model
        
        model.fit(X_train_scaled, y_train)  # Train model
        
        # Validate model
        y_pred = model.predict(X_val_scaled)  # Make predictions
        accuracy = accuracy_score(y_val, y_pred)  # Calculate accuracy
        print(f"Validation accuracy: {accuracy:.4f}")  # Print accuracy
        
        # Store model, scaler, and config
        models.append(model)  # Add model to list
        scalers.append(scaler)  # Add scaler to list
        configs.append(config)  # Add config to list
    
    print(f"\nTrained {len(models)} ensemble models")  # Print completion message
    
    return models, scalers, configs  # Return trained components


def save_ensemble(models, scalers, configs):
    """
    Save ensemble models and scalers to disk.
    
    Args:
        models: List of trained models
        scalers: List of trained scalers
        configs: List of model configurations
    """
    print("Saving ensemble models...")  # Print progress message
    
    # Create ensemble directory
    ensemble_dir = os.path.join(MODELS_DIR, "ensemble")  # Get ensemble directory path
    os.makedirs(ensemble_dir, exist_ok=True)  # Create directory
    
    # Save each model and scaler
    for i, (model, scaler, config) in enumerate(zip(models, scalers, configs)):  # Iterate through models
        model_path = os.path.join(ensemble_dir, f"model_{config['name']}.pkl")  # Model path
        scaler_path = os.path.join(ensemble_dir, f"scaler_{config['name']}.pkl")  # Scaler path
        
        with open(model_path, 'wb') as f:  # Open model file
            pickle.dump(model, f)  # Save model
        
        with open(scaler_path, 'wb') as f:  # Open scaler file
            pickle.dump(scaler, f)  # Save scaler
        
        print(f"Saved: {config['name']}")  # Print save confirmation
    
    # Save configuration list
    config_path = os.path.join(ensemble_dir, "configs.pkl")  # Config path
    with open(config_path, 'wb') as f:  # Open config file
        pickle.dump(configs, f)  # Save configurations
    
    print(f"Ensemble saved to: {ensemble_dir}")  # Print completion message


def load_ensemble():
    """
    Load ensemble models and scalers from disk.
    
    Returns:
        models: List of loaded models
        scalers: List of loaded scalers
        configs: List of model configurations
    """
    print("Loading ensemble models...")  # Print progress message
    
    # Load configurations
    ensemble_dir = os.path.join(MODELS_DIR, "ensemble")  # Get ensemble directory
    config_path = os.path.join(ensemble_dir, "configs.pkl")  # Config path
    
    with open(config_path, 'rb') as f:  # Open config file
        configs = pickle.load(f)  # Load configurations
    
    # Load models and scalers
    models = []  # Initialize models list
    scalers = []  # Initialize scalers list
    
    for config in configs:  # Iterate through configurations
        model_path = os.path.join(ensemble_dir, f"model_{config['name']}.pkl")  # Model path
        scaler_path = os.path.join(ensemble_dir, f"scaler_{config['name']}.pkl")  # Scaler path
        
        with open(model_path, 'rb') as f:  # Open model file
            model = pickle.load(f)  # Load model
        
        with open(scaler_path, 'rb') as f:  # Open scaler file
            scaler = pickle.load(f)  # Load scaler
        
        models.append(model)  # Add model to list
        scalers.append(scaler)  # Add scaler to list
    
    print(f"Loaded {len(models)} ensemble models")  # Print completion message
    
    return models, scalers, configs  # Return loaded components


def predict_ensemble(models, scalers, X):
    """
    Make ensemble predictions using voting.
    
    Args:
        models: List of trained models
        scalers: List of trained scalers
        X: Input features
        
    Returns:
        predictions: Ensemble predictions
        probabilities: Average probabilities
    """
    all_predictions = []  # Initialize predictions list
    all_probabilities = []  # Initialize probabilities list
    
    # Make predictions with each model
    for model, scaler in zip(models, scalers):  # Iterate through models
        X_scaled = scaler.transform(X)  # Scale features
        pred = model.predict(X_scaled)  # Make predictions
        prob = model.predict_proba(X_scaled)  # Get probabilities
        
        all_predictions.append(pred)  # Add predictions
        all_probabilities.append(prob)  # Add probabilities
    
    # Majority voting for predictions
    all_predictions = np.array(all_predictions)  # Convert to array
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions
    )  # Majority vote
    
    # Average probabilities
    avg_probabilities = np.mean(all_probabilities, axis=0)  # Average probabilities
    
    return ensemble_pred, avg_probabilities  # Return ensemble results


def evaluate_ensemble():
    """
    Evaluate ensemble model performance.
    """
    print("Evaluating ensemble...")  # Print progress message
    
    # Load ensemble
    models, scalers, configs = load_ensemble()  # Load trained models
    
    # Load test dataset
    from dataset import load_test_dataset  # Import test dataset function
    X_test, y_test = load_test_dataset()  # Load test data
    
    # Make ensemble predictions
    y_pred, probabilities = predict_ensemble(models, scalers, X_test)  # Predict
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Ensemble accuracy: {accuracy:.4f}")  # Print accuracy
    
    # Individual model accuracies
    print("\nIndividual model accuracies:")
    for i, config in enumerate(configs):  # Iterate through configs
        X_scaled = scalers[i].transform(X_test)  # Scale features
        pred = models[i].predict(X_scaled)  # Make predictions
        acc = accuracy_score(y_test, pred)  # Calculate accuracy
        print(f"{config['name']}: {acc:.4f}")  # Print individual accuracy
    
    return accuracy  # Return ensemble accuracy


if __name__ == "__main__":  # Check if script is run directly
    # Train and save ensemble
    models, scalers, configs = train_ensemble_models()  # Train models
    save_ensemble(models, scalers, configs)  # Save models
    
    # Evaluate ensemble
    ensemble_accuracy = evaluate_ensemble()  # Evaluate performance
    print(f"\nFinal ensemble accuracy: {ensemble_accuracy:.4f}")  # Print final accuracy
