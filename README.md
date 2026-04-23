# Computer Vision Project

A comprehensive computer vision project for image classification using HOG (Histogram of Oriented Gradients) features and SVM classifiers.

## Project Structure

```
current folder/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── class_0/
│   │   │   └── class_1/
│   │   └── test/
│   │       ├── class_0/
│   │       └── class_1/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hog_visualization.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── svm_model.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Features

- **Data Preprocessing**: Image loading, resizing, normalization, and augmentation
- **Feature Extraction**: HOG features, color histograms, and LBP (Local Binary Patterns)
- **Model Training**: SVM with hyperparameter tuning, cross-validation
- **Evaluation**: Comprehensive metrics, confusion matrices, ROC curves
- **Visualization**: Feature visualizations, learning curves, error analysis
- **Prediction**: Single image and batch prediction capabilities

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your images in the following structure:
```
data/raw/train/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class_1/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Training

Run the training pipeline:
```python
from src.train import train_and_evaluate_pipeline

# Train with grid search
model, label_encoder, results = train_and_evaluate_pipeline(use_grid_search=True)

# Train with simple SVM
model, label_encoder, results = train_and_evaluate_pipeline(use_grid_search=False)
```

### Prediction

Make predictions on new images:
```python
from src.predict import predict_single_image, predict_directory

# Single image prediction
result = predict_single_image("path/to/image.jpg", return_probabilities=True)
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']}")

# Batch prediction
results = predict_directory("path/to/images/", save_predictions=True)
```

### Evaluation

Evaluate a trained model:
```python
from src.evaluate import evaluate_saved_model

results = evaluate_saved_model()
```

## Notebooks

- **01_data_exploration.ipynb**: Explore dataset structure, visualize samples, analyze class distribution
- **02_hog_visualization.ipynb**: HOG feature extraction and visualization
- **03_model_experiments.ipynb**: Model comparison, hyperparameter tuning, performance analysis

## Configuration

Modify `src/config.py` to adjust:
- HOG parameters (orientations, pixels per cell, cells per block)
- SVM parameters (C, kernel, gamma)
- Image size and preprocessing settings
- File paths

## Models

The project supports multiple classifiers:
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes

## Feature Types

- **HOG**: Histogram of Oriented Gradients for shape/edge information
- **Color Histogram**: RGB color distribution
- **LBP**: Local Binary Patterns for texture information
- **Combined**: Concatenation of all feature types

## Output Files

- **models/**: Trained models and label encoders
- **outputs/figures/**: Plots and visualizations
- **outputs/metrics/**: Evaluation metrics and reports
- **outputs/predictions/**: Prediction results

## Performance Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves (for binary classification)
- Cross-validation scores
- Learning curves

## Dependencies

- numpy, pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- opencv-python: Image processing
- scikit-image: Feature extraction
- matplotlib, seaborn: Visualization
- imbalanced-learn: Data balancing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with scikit-learn and OpenCV
- HOG features implementation inspired by skimage
- Project structure follows best practices for ML projects
