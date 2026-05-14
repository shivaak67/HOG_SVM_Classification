"""FastAPI backend for the HOG + SVM cat vs dog classifier."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_utils import ModelNotFoundError, load_artifacts, predict_image


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the API starts."""
    try:
        app.state.model, app.state.scaler = load_artifacts()
    except ModelNotFoundError as error:
        app.state.model = None
        app.state.scaler = None
        app.state.startup_error = str(error)
    yield


app = FastAPI(
    title="HOG SVM Cat vs Dog API",
    description="Predict whether an uploaded image is a cat or dog.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Simple health check for the API."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept one image upload and return the model prediction."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, BMP, or WebP image.",
        )

    if app.state.model is None or app.state.scaler is None:
        raise HTTPException(
            status_code=500,
            detail=getattr(
                app.state,
                "startup_error",
                "Model files are not loaded.",
            ),
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return predict_image(image_bytes, app.state.model, app.state.scaler)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check that the model, scaler, and HOG settings match training.",
        ) from error
