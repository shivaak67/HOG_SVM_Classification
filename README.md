# HOG SVM Cat vs Dog Classification

A computer vision project for classifying cat and dog images using HOG (Histogram of Oriented Gradients) features and an SVM classifier. The project includes training/evaluation scripts, exploratory notebooks, and a full-stack web app for uploading an image and getting a prediction.

## Project Structure

```text
HOG_SVM_Classification/
|
|-- data/
|   |-- raw/
|   |   |-- train/
|   |   |   |-- class_0/
|   |   |   `-- class_1/
|   |   `-- test/
|   |       |-- class_0/
|   |       `-- class_1/
|   `-- processed/
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_hog_visualization.ipynb
|   `-- 03_model_experiments.ipynb
|
|-- src/
|   |-- config.py
|   |-- dataset.py
|   |-- train.py
|   |-- evaluate.py
|   `-- ensemble.py
|
|-- backend/
|   |-- app.py
|   |-- model_utils.py
|   |-- requirements.txt
|   `-- models/
|
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- vite.config.js
|
|-- models/
|-- outputs/
|-- requirements.txt
|-- WEB_APP_SETUP.md
`-- README.md
```

## Core ML Features

- HOG feature extraction with configurable image size and HOG parameters
- SVM model training with feature scaling
- Evaluation with accuracy, precision, recall, F1 score, and confusion matrix
- Ensemble/model experiment scripts
- Notebooks for data exploration, HOG visualization, and model comparison

## Web App

The project includes a full-stack cat vs dog prediction website.

- Frontend: React, Vite, Tailwind CSS
- Backend: FastAPI
- ML inference: OpenCV preprocessing, scikit-image HOG extraction, scikit-learn scaler and SVM model
- Endpoint: `POST /predict`

The website lets a user drag and drop or upload an image, preview it, click Predict, and see the predicted class with a confidence score when the model supports probabilities.

## Project Screenshots
<img width="1602" height="869" alt="Screenshot 2026-05-16 171107" src="https://github.com/user-attachments/assets/6944c69e-62f6-4359-9f8f-10b44bc49b66" />
<img width="1639" height="879" alt="Screenshot 2026-05-16 171135" src="https://github.com/user-attachments/assets/8a89c7e7-7595-4528-9f45-0015b9c696d4" />
<img width="1636" height="852" alt="Screenshot 2026-05-16 171208" src="https://github.com/user-attachments/assets/cf1a3f37-a77d-43e1-b9d0-f433b129a195" />




## Installation

Install the main ML project dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place images in this format:

```text
data/raw/train/
|-- class_0/
|   |-- image1.jpg
|   `-- image2.jpg
`-- class_1/
    |-- image1.jpg
    `-- image2.jpg
```

The current web app label mapping assumes:

```python
0 -> cat
1 -> dog
```

If your folders use the opposite order, update `CLASS_NAMES` in `backend/model_utils.py`.

## Training

Run the training script from the project root:

```bash
python src/train.py
```

This trains the SVM model using the settings in `src/config.py` and saves model artifacts under `models/`.

## Evaluation

Run:

```bash
python src/evaluate.py
```

The evaluation pipeline loads the saved model and scaler, preprocesses test images, extracts HOG features, and reports classification metrics.

## Web App Setup

### Model Files

The FastAPI backend expects:

```text
backend/models/svm_model.pkl
backend/models/scaler.pkl
```

For local development, the backend also falls back to the existing project model files:

```text
models/hog_svm_model.pkl
models/scaler.pkl
```

Large `.pkl` model files are ignored by Git.

### Run Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Backend URL:

```text
http://localhost:8000
```

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

Example response:

```json
{
  "prediction": "cat",
  "confidence": 0.87
}
```

### Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Inference Configuration

Edit `backend/model_utils.py` to change web app inference settings:

```python
IMAGE_SIZE = (128, 128)

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}
```

These values must match the settings used during model training.

## Notebooks

- `01_data_exploration.ipynb`: dataset structure, sample images, and class distribution
- `02_hog_visualization.ipynb`: original image, grayscale image, HOG visualization, and feature preview
- `03_model_experiments.ipynb`: simple SVM experiment comparison and validation metrics

## Important Notes

- Do not commit trained `.pkl` model files to GitHub because they are large.
- Do not retrain the model from the web app. The backend only loads existing model artifacts.
- Keep `src/config.py` and `backend/model_utils.py` aligned when changing image size or HOG parameters.

## Dependencies

- Python: NumPy, pandas, OpenCV, scikit-image, scikit-learn, matplotlib, seaborn
- Backend: FastAPI, Uvicorn, python-multipart
- Frontend: React, Vite, Tailwind CSS

## Acknowledgments

- Built with scikit-learn, scikit-image, OpenCV, FastAPI, React, and Tailwind CSS
