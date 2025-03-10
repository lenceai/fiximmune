"""
REST API for serving treatment recommendation model predictions.

This module implements a FastAPI service for the machine learning model,
allowing it to be used by the main FixImmune application.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
from ..inference.predict import (
    get_treatment_recommendations,
    load_latest_model,
    load_model_by_id
)
from ..data.download import DEFAULT_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FixImmune Treatment Recommendation API",
    description="API for getting personalized treatment recommendations for autoimmune patients",
    version="0.1.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODELS_DIR = DEFAULT_DATA_DIR / "models"


# Pydantic models for request/response validation
class PatientData(BaseModel):
    """Patient data for recommendation requests."""
    id: Optional[str] = None
    age: int
    gender: str
    weight: Optional[float] = None
    height: Optional[float] = None
    disease_type: str
    symptom_severity: int
    diagnosis_date: Optional[str] = None
    medical_history: Optional[str] = None
    allergies: Optional[str] = None
    previous_treatments: Optional[List[str]] = None


class TreatmentPrediction(BaseModel):
    """Treatment prediction result."""
    treatment_id: str
    treatment_name: str
    predicted_effectiveness: float
    confidence: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Response model for treatment recommendations."""
    patient_id: Optional[str] = None
    model_id: str
    include_experimental: bool
    predictions: List[TreatmentPrediction]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


# API endpoints
@app.get("/")
async def root():
    """Root endpoint that provides basic API information."""
    return {
        "name": "FixImmune Treatment Recommendation API",
        "version": "0.1.1",
        "status": "active",
        "documentation": "/docs",
    }


@app.get("/api/v1/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List available trained models."""
    try:
        # Find all model directories
        model_dirs = [d for d in MODELS_DIR.glob("model_*") if d.is_dir()]
        
        if not model_dirs:
            return []
        
        # Get model information
        models_info = []
        for model_dir in model_dirs:
            model_id = model_dir.name.split("_")[1]
            
            # Try to load model info
            info_path = model_dir / "model_info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    model_info = json.load(f)
                    
                    # Add model ID if not present
                    if "timestamp" not in model_info:
                        model_info["timestamp"] = model_id
                    
                    models_info.append(model_info)
            else:
                # Create minimal info
                models_info.append({
                    "timestamp": model_id,
                    "path": str(model_dir)
                })
        
        # Sort by timestamp (descending)
        models_info.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return models_info
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@app.get("/api/v1/models/{model_id}", response_model=Dict[str, Any])
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    try:
        # Find model directory
        model_dir = MODELS_DIR / f"model_{model_id}"
        
        if not model_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID {model_id} not found"
            )
        
        # Try to load model info
        info_path = model_dir / "model_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                model_info = json.load(f)
                
                # Add model ID if not present
                if "timestamp" not in model_info:
                    model_info["timestamp"] = model_id
                
                return model_info
        else:
            # Create minimal info
            return {
                "timestamp": model_id,
                "path": str(model_dir)
            }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@app.post(
    "/api/v1/recommendations",
    response_model=RecommendationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def get_recommendations(
    patient_data: PatientData,
    model_id: Optional[str] = Query(None, description="ID of model to use (uses latest if not provided)"),
    include_experimental: bool = Query(False, description="Whether to include experimental treatments"),
    limit: int = Query(5, description="Maximum number of recommendations to return")
):
    """
    Get personalized treatment recommendations for a patient.
    
    This endpoint analyzes patient data and returns a ranked list of recommended treatments.
    """
    try:
        # Convert patient data to dictionary
        patient_dict = patient_data.dict()
        
        # Get recommendations
        recommendations = get_treatment_recommendations(
            patient_dict,
            model_id=model_id,
            include_experimental=include_experimental,
            limit=limit
        )
        
        # Check for errors
        if "error" in recommendations:
            logger.error(f"Error in recommendations: {recommendations['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=recommendations["error"]
            )
        
        # Convert to response model
        response = RecommendationResponse(
            patient_id=recommendations.get("patient_id"),
            model_id=recommendations.get("model_id", "unknown"),
            include_experimental=recommendations.get("include_experimental", include_experimental),
            predictions=[
                TreatmentPrediction(
                    treatment_id=pred.get("treatment_id", "unknown"),
                    treatment_name=pred.get("treatment_name", "Unknown"),
                    predicted_effectiveness=pred.get("predicted_effectiveness", 0.0),
                    confidence=pred.get("confidence")
                )
                for pred in recommendations.get("predictions", [])
            ]
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )


@app.post(
    "/api/v1/predictions/single",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict_single_treatment(
    patient_data: Dict[str, Any] = Body(..., description="Patient data including treatment information")
):
    """
    Predict effectiveness of a single treatment for a patient.
    
    This endpoint requires both patient data and treatment data in a single request.
    """
    try:
        # Ensure treatment_id is provided
        if "treatment_id" not in patient_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Treatment ID is required"
            )
        
        # Use the latest model for prediction
        model, model_info = load_latest_model()
        
        # Get feature metadata
        feature_metadata = model_info.get("feature_metadata", {})
        if not feature_metadata:
            # Try to load from file
            model_dir = MODELS_DIR / f"model_{model_info.get('timestamp', '')}"
            metadata_path = model_dir / "feature_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    feature_metadata = json.load(f)
        
        # Get paths to encoders/scalers
        model_dir = MODELS_DIR / f"model_{model_info.get('timestamp', '')}"
        encoder_path = model_dir / "categorical_encoder.pkl"
        scaler_path = model_dir / "numeric_scaler.pkl"
        
        # Import prediction function directly
        from ..inference.predict import predict_treatment_effectiveness
        
        # Make prediction
        result = predict_treatment_effectiveness(
            model, 
            patient_data, 
            feature_metadata,
            encoder_path=encoder_path,
            scaler_path=scaler_path
        )
        
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error predicting treatment effectiveness: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting treatment effectiveness: {str(e)}"
        )


def start_model_api(host="0.0.0.0", port=8001):
    """
    Start the model API server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    import uvicorn
    
    # Start server
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    # Start the API server
    start_model_api() 