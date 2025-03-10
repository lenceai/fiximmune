"""
Model inference for treatment recommendations.

This module handles inference with trained models to generate
personalized treatment recommendations for autoimmune patients.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union

# Local imports
from ..data.download import DEFAULT_DATA_DIR
from ..features.feature_engineering import FEATURES_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = DEFAULT_DATA_DIR / "models"


def load_latest_model(models_dir: Path = MODELS_DIR) -> Tuple[Any, Dict[str, Any]]:
    """
    Load the most recently trained model.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        Tuple of (model, model_info)
    """
    logger.info(f"Loading latest model from {models_dir}")
    
    # Find all model directories
    model_dirs = [d for d in models_dir.glob("model_*") if d.is_dir()]
    
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {models_dir}")
    
    # Sort by timestamp (directory name)
    latest_model_dir = sorted(model_dirs)[-1]
    
    # Load model
    model_path = latest_model_dir / "model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load model info
    info_path = latest_model_dir / "model_info.json"
    
    if not info_path.exists():
        # Create minimal info if file doesn't exist
        logger.warning(f"Model info file not found at {info_path}")
        model_info = {"timestamp": latest_model_dir.name.split("_")[1]}
    else:
        with open(info_path, "r") as f:
            model_info = json.load(f)
    
    logger.info(f"Loaded model from {latest_model_dir}")
    
    return model, model_info


def load_model_by_id(model_id: str, models_dir: Path = MODELS_DIR) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a specific model by its ID (timestamp).
    
    Args:
        model_id: Model ID (timestamp)
        models_dir: Directory containing trained models
        
    Returns:
        Tuple of (model, model_info)
    """
    logger.info(f"Loading model {model_id}")
    
    # Find model directory
    model_dir = models_dir / f"model_{model_id}"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load model
    model_path = model_dir / "model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load model info
    info_path = model_dir / "model_info.json"
    
    if not info_path.exists():
        # Create minimal info if file doesn't exist
        logger.warning(f"Model info file not found at {info_path}")
        model_info = {"timestamp": model_id}
    else:
        with open(info_path, "r") as f:
            model_info = json.load(f)
    
    logger.info(f"Loaded model from {model_dir}")
    
    return model, model_info


def prepare_data_for_prediction(
    patient_data: Dict[str, Any],
    feature_metadata: Dict[str, Any],
    encoder_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Prepare patient data for prediction.
    
    Args:
        patient_data: Dictionary of patient data
        feature_metadata: Feature metadata from the model
        encoder_path: Path to categorical encoder (optional)
        scaler_path: Path to numeric scaler (optional)
        
    Returns:
        DataFrame ready for prediction
    """
    logger.info("Preparing data for prediction")
    
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Apply preprocessing steps similar to those used during training
    # This would include handling missing values, encoding categorical variables, etc.
    
    # For a real implementation, you should apply the same preprocessing pipeline
    # that was used during training. This may involve loading saved encoders/scalers.
    
    # Load encoders if provided
    encoder = None
    if encoder_path and encoder_path.exists():
        try:
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
                logger.info(f"Loaded encoder from {encoder_path}")
        except Exception as e:
            logger.warning(f"Failed to load encoder: {e}")
    
    # Load scalers if provided
    scaler = None
    if scaler_path and scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
    
    # Apply preprocessing steps
    # This is a simplified version - in a real application, you would apply
    # the full preprocessing pipeline used during training
    
    # Basic preprocessing to handle missing values
    for col in patient_df.columns:
        if col in feature_metadata.get("feature_types", {}).get("numeric", []):
            patient_df[col] = patient_df[col].fillna(0)
        elif col in feature_metadata.get("feature_types", {}).get("categorical", []):
            patient_df[col] = patient_df[col].fillna("Unknown")
    
    # Apply encoding to categorical features if encoder is available
    if encoder:
        # Apply encoding based on type of encoder
        if hasattr(encoder, "transform"):
            # Assume it's a scikit-learn transformer
            cat_cols = feature_metadata.get("feature_types", {}).get("categorical", [])
            if cat_cols:
                try:
                    encoded_features = encoder.transform(patient_df[cat_cols])
                    
                    # If encoder is OneHotEncoder, get feature names
                    if hasattr(encoder, "get_feature_names_out"):
                        encoded_cols = encoder.get_feature_names_out(cat_cols)
                        encoded_df = pd.DataFrame(encoded_features, columns=encoded_cols, index=patient_df.index)
                        
                        # Drop original categorical columns and add encoded ones
                        patient_df = patient_df.drop(columns=cat_cols)
                        patient_df = pd.concat([patient_df, encoded_df], axis=1)
                except Exception as e:
                    logger.warning(f"Failed to apply encoding: {e}")
    
    # Apply scaling to numeric features if scaler is available
    if scaler:
        num_cols = feature_metadata.get("feature_types", {}).get("numeric", [])
        if num_cols:
            try:
                patient_df[num_cols] = scaler.transform(patient_df[num_cols])
            except Exception as e:
                logger.warning(f"Failed to apply scaling: {e}")
    
    # Ensure we have all required columns for prediction
    required_columns = feature_metadata.get("required_columns", [])
    if required_columns:
        missing_cols = [col for col in required_columns if col not in patient_df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}. Will add with default values.")
            for col in missing_cols:
                patient_df[col] = 0  # Add default values
    
    logger.info(f"Prepared data for prediction: {patient_df.shape}")
    
    return patient_df


def predict_treatment_effectiveness(
    model: Any,
    patient_data: Dict[str, Any],
    feature_metadata: Dict[str, Any],
    treatment_data: Optional[pd.DataFrame] = None,
    encoder_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Predict treatment effectiveness for a patient.
    
    Args:
        model: Trained model
        patient_data: Dictionary of patient data
        feature_metadata: Feature metadata from the model
        treatment_data: DataFrame of available treatments (optional)
        encoder_path: Path to categorical encoder (optional)
        scaler_path: Path to numeric scaler (optional)
        
    Returns:
        Dictionary of prediction results
    """
    logger.info("Predicting treatment effectiveness")
    
    # Prepare data for prediction
    patient_df = prepare_data_for_prediction(
        patient_data, feature_metadata, encoder_path, scaler_path
    )
    
    # If treatment_data is not provided, we can only make a single prediction
    if treatment_data is None:
        # Make prediction
        try:
            prediction = model.predict(patient_df)[0]
            
            # For classification models, get probability
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(patient_df)[0]
                
                # Format prediction result
                result = {
                    "predicted_effectiveness": float(prediction),
                    "confidence": float(probabilities.max()),
                    "treatment_id": patient_data.get("treatment_id", "unknown")
                }
            else:
                # For regression models
                result = {
                    "predicted_effectiveness": float(prediction),
                    "treatment_id": patient_data.get("treatment_id", "unknown")
                }
            
            return {"predictions": [result]}
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    # If treatment_data is provided, make predictions for each treatment
    else:
        predictions = []
        
        for _, treatment in treatment_data.iterrows():
            # Create a copy of patient data for each treatment
            treatment_patient_data = patient_data.copy()
            
            # Add treatment data
            for col, value in treatment.items():
                treatment_patient_data[col] = value
            
            # Prepare data for this treatment
            treatment_patient_df = prepare_data_for_prediction(
                treatment_patient_data, feature_metadata, encoder_path, scaler_path
            )
            
            # Make prediction
            try:
                prediction = model.predict(treatment_patient_df)[0]
                
                # For classification models, get probability
                confidence = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(treatment_patient_df)[0]
                    confidence = float(probabilities.max())
                
                # Format prediction result
                result = {
                    "treatment_id": treatment.get("id", "unknown"),
                    "treatment_name": treatment.get("name", "Unknown"),
                    "predicted_effectiveness": float(prediction),
                }
                
                if confidence is not None:
                    result["confidence"] = confidence
                
                predictions.append(result)
                
            except Exception as e:
                logger.error(f"Prediction failed for treatment {treatment.get('name', 'unknown')}: {e}")
        
        # Sort predictions by effectiveness (descending)
        predictions.sort(key=lambda x: x["predicted_effectiveness"], reverse=True)
        
        return {"predictions": predictions}


def get_treatment_recommendations(
    patient_data: Dict[str, Any],
    model_id: Optional[str] = None,
    include_experimental: bool = False,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get treatment recommendations for a patient.
    
    Args:
        patient_data: Dictionary of patient data
        model_id: ID of model to use (optional, uses latest if not provided)
        include_experimental: Whether to include experimental treatments
        limit: Maximum number of recommendations to return
        
    Returns:
        Dictionary containing treatment recommendations
    """
    logger.info(f"Getting treatment recommendations for patient with experimental={include_experimental}")
    
    try:
        # Load model
        if model_id:
            model, model_info = load_model_by_id(model_id)
        else:
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
        
        # Load available treatments from database or file
        # For demo purposes, we'll create some fake treatments
        treatments = []
        for i in range(10):
            treatment = {
                "id": f"T{i+1}",
                "name": f"Treatment {i+1}",
                "type": "medication" if i < 7 else "biologic",
                "is_experimental": i >= 8,
                "avg_effectiveness": (10 - i) * 0.4  # Dummy effectiveness value
            }
            treatments.append(treatment)
        
        # Create treatments DataFrame
        treatments_df = pd.DataFrame(treatments)
        
        # Filter experimental treatments if not included
        if not include_experimental:
            treatments_df = treatments_df[~treatments_df["is_experimental"]]
        
        # Make predictions for each treatment
        result = predict_treatment_effectiveness(
            model,
            patient_data,
            feature_metadata,
            treatments_df,
            encoder_path,
            scaler_path
        )
        
        # Limit number of recommendations
        if "predictions" in result:
            result["predictions"] = result["predictions"][:limit]
        
        # Add metadata to result
        result["model_id"] = model_info.get("timestamp", "unknown")
        result["patient_id"] = patient_data.get("id", "unknown")
        result["include_experimental"] = include_experimental
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting treatment recommendations: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage with dummy data
    patient_data = {
        "id": "P12345",
        "age": 45,
        "gender": "Female",
        "weight": 65.5,
        "height": 165.0,
        "disease_type": "rheumatoid_arthritis",
        "symptom_severity": 3,
        "diagnosis_date": "2020-05-15"
    }
    
    try:
        # Get recommendations
        recommendations = get_treatment_recommendations(
            patient_data,
            include_experimental=True,
            limit=3
        )
        
        # Print recommendations
        if "error" in recommendations:
            logger.error(f"Error: {recommendations['error']}")
        else:
            logger.info(f"Got {len(recommendations['predictions'])} recommendations")
            
            for i, rec in enumerate(recommendations['predictions']):
                logger.info(f"Recommendation {i+1}: {rec['treatment_name']} - Effectiveness: {rec['predicted_effectiveness']:.2f}")
    
    except Exception as e:
        logger.error(f"Error in example: {e}") 