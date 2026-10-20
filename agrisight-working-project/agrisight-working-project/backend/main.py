from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# Support both `uvicorn backend.main:app` from the project root and
# `uvicorn main:app` from inside backend/.
try:
    from .agrisight import gradcam_utils as gu
    from .agrisight import models
except ImportError:  # pragma: no cover - exercised by the backend-directory launch
    from agrisight import gradcam_utils as gu
    from agrisight import models

MAX_IMAGE_BYTES = int(os.getenv("AGRISIGHT_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "mobilenet_field.pt"
MODEL_CHECKPOINT = os.getenv("AGRISIGHT_MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)).strip()
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("AGRISIGHT_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(title="AgriSight inference service", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)


def _checkpoint_path() -> Path:
    if not MODEL_CHECKPOINT:
        raise HTTPException(
            status_code=503,
            detail="MODEL_CHECKPOINT_NOT_CONFIGURED: Place a trained .pt checkpoint at backend/models/mobilenet_field.pt or set AGRISIGHT_MODEL_CHECKPOINT.",
        )
    path = Path(MODEL_CHECKPOINT).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="MODEL_CHECKPOINT_NOT_FOUND: The configured model checkpoint file does not exist.",
        )
    return path


@lru_cache(maxsize=1)
def _load_model(checkpoint: str):
    """Load the real trained checkpoint once per running worker."""
    return models.load_checkpoint(checkpoint, "cpu")


def _pretty(label: str) -> str:
    return label.replace("___", ": ").replace("_", " ")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "AgriSight inference API", "health": "/health", "prediction": "/predict"}


@app.get("/health")
def health() -> dict[str, str | bool]:
    configured = bool(MODEL_CHECKPOINT) and Path(MODEL_CHECKPOINT).expanduser().is_file()
    return {
        "ok": True,
        "model_configured": configured,
        "model_status": "ready" if configured else "checkpoint_required",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")

    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image is too large (10 MiB maximum by default).")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    checkpoint = _checkpoint_path()
    try:
        model, metadata = _load_model(str(checkpoint))
        probabilities, _ = gu.predict(model, image)
    except HTTPException:
        raise
    except Exception as exc:
        # Avoid exposing internal paths/tracebacks to the browser; server logs retain details.
        print(f"AgriSight model inference failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="MODEL_INFERENCE_FAILED: The trained model could not run. Check the checkpoint and backend logs.")

    classes = metadata.get("classes")
    model_name = metadata.get("model")
    if not isinstance(classes, list) or len(classes) != len(probabilities) or not isinstance(model_name, str):
        raise HTTPException(status_code=503, detail="MODEL_CHECKPOINT_INVALID: The checkpoint must contain model, classes, and state_dict.")

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {"label": _pretty(str(classes[int(index)])), "confidence": float(probabilities[index])}
        for index in top_indices
    ]

    # Grad-CAM is an optional explanation. It must never prevent a valid model
    # prediction from being returned when the visualization dependency fails.
    background_reliance = 0.0
    gradcam_available = False
    try:
        overlay, cam = gu.gradcam_overlay(model, model_name, image, int(top_indices[0]))
        if overlay is not None and cam is not None:
            background_reliance = float(gu.background_reliance(cam))
            gradcam_available = True
    except Exception as exc:
        print(f"AgriSight Grad-CAM unavailable: {type(exc).__name__}: {exc}")

    return {
        "prediction": top_predictions[0]["label"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "background_reliance": background_reliance,
        "model": model_name,
        "classes_covered": len(classes),
        "gradcam_available": gradcam_available,
    }
