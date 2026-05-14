# HOG + SVM Cat vs Dog Web App

This project adds a React frontend and FastAPI backend around the trained HOG + SVM classifier. It does not retrain the model.

## Model Files

Place your trained files here:

```text
backend/models/svm_model.pkl
backend/models/scaler.pkl
```

Your current project model files are named:

```text
models/hog_svm_model.pkl
models/scaler.pkl
```

Copy them into `backend/models/` and rename `hog_svm_model.pkl` to `svm_model.pkl`.

The backend also falls back to the existing project files in `models/`, so you can run it without copying the large model file while working locally.

## Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The API runs at `http://localhost:8000`.

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

Expected response:

```json
{
  "prediction": "cat",
  "confidence": 0.87
}
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The React app runs at `http://localhost:5173`.

## Configuration

Edit the inference settings in `backend/model_utils.py`:

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

These values must match the settings used when the SVM model and scaler were trained.
