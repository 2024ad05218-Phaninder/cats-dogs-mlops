"""
FastAPI Inference Service for Cats vs Dogs Classification
REST API with health check and prediction endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
from PIL import Image
import io
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inference import CatsDogsClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="REST API for binary image classification (Cats vs Dogs)",
    version="1.0.0"
)

# Global model instance
classifier = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    class_index: int


class PredictionRequest(BaseModel):
    """Prediction request (for JSON input)"""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global classifier

    try:
        model_path = "models/best_model.h5"
        logger.info(f"Loading model from {model_path}")

        classifier = CatsDogsClassifier(model_path)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        classifier = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        Health status and model information
    """
    try:
        model_loaded = classifier is not None

        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image

    Args:
        file: Image file (JPG, PNG, etc.)

    Returns:
        Prediction with class, confidence, and probabilities
    """
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Predict
        result = classifier.predict(image)

        return PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            class_index=result['class_index']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """
    Get model information

    Returns:
        Model architecture details
    """
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "model_path": str(classifier.model_path),
            "image_size": classifier.image_size,
            "classes": classifier.class_names,
            "model_parameters": int(classifier.model.count_params()),
            "num_layers": len(classifier.model.layers)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")


@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image file)",
            "model_info": "/model-info",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    """Swagger UI documentation"""
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
