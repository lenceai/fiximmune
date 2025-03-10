"""
Integration of ML model API with main FixImmune application.

This module provides functions to integrate the ML model API with the
main FixImmune application for treatment recommendations.
"""

import os
import logging
import requests
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Default API URL
DEFAULT_ML_API_URL = os.getenv("ML_API_URL", "http://localhost:8001")


class MLApiClient:
    """Client for interacting with the ML model API."""
    
    def __init__(self, api_url: str = DEFAULT_ML_API_URL):
        """
        Initialize ML API client.
        
        Args:
            api_url: URL of the ML API service
        """
        self.api_url = api_url
        logger.info(f"Initialized ML API client with URL: {api_url}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available trained models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = requests.get(f"{self.api_url}/api/v1/models")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model to get information for
            
        Returns:
            Dictionary containing model information
        """
        try:
            response = requests.get(f"{self.api_url}/api/v1/models/{model_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def get_recommendations(
        self,
        patient_data: Dict[str, Any],
        model_id: Optional[str] = None,
        include_experimental: bool = False,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get treatment recommendations for a patient.
        
        Args:
            patient_data: Dictionary of patient data
            model_id: ID of model to use (optional)
            include_experimental: Whether to include experimental treatments
            limit: Maximum number of recommendations to return
            
        Returns:
            Dictionary containing treatment recommendations
        """
        try:
            # Build request URL with query parameters
            url = f"{self.api_url}/api/v1/recommendations"
            params = {}
            
            if model_id:
                params["model_id"] = model_id
            
            params["include_experimental"] = str(include_experimental).lower()
            params["limit"] = str(limit)
            
            # Make API request
            response = requests.post(url, params=params, json=patient_data)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting recommendations: {e}")
            
            # Try to parse error message from response
            error_detail = str(e)
            try:
                if hasattr(e, "response") and e.response is not None:
                    error_data = e.response.json()
                    if "detail" in error_data:
                        error_detail = error_data["detail"]
            except:
                pass
            
            return {"error": error_detail}
    
    def predict_single_treatment(
        self,
        patient_data: Dict[str, Any],
        treatment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict effectiveness of a single treatment for a patient.
        
        Args:
            patient_data: Dictionary of patient data
            treatment_data: Dictionary of treatment data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Combine patient and treatment data
            combined_data = {
                **patient_data,
                **treatment_data,
                "treatment_id": treatment_data.get("id", "unknown")
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_url}/api/v1/predictions/single",
                json=combined_data
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error predicting treatment effectiveness: {e}")
            
            # Try to parse error message from response
            error_detail = str(e)
            try:
                if hasattr(e, "response") and e.response is not None:
                    error_data = e.response.json()
                    if "detail" in error_data:
                        error_detail = error_data["detail"]
            except:
                pass
            
            return {"error": error_detail}


def get_ml_client() -> MLApiClient:
    """
    Get ML API client instance.
    
    Returns:
        ML API client instance
    """
    return MLApiClient()


def get_treatment_recommendations_for_patient(
    patient_id: str,
    patient_data: Dict[str, Any],
    include_experimental: bool = False,
    model_id: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get treatment recommendations for a patient.
    
    This is a convenience function for integration with the main API.
    
    Args:
        patient_id: ID of the patient
        patient_data: Patient data
        include_experimental: Whether to include experimental treatments
        model_id: ID of model to use (optional)
        limit: Maximum number of recommendations to return
        
    Returns:
        Dictionary containing treatment recommendations
    """
    # Ensure patient_id is set
    if "id" not in patient_data:
        patient_data["id"] = patient_id
    
    # Get client and request recommendations
    client = get_ml_client()
    return client.get_recommendations(
        patient_data,
        model_id=model_id,
        include_experimental=include_experimental,
        limit=limit
    )


def check_ml_api_status() -> bool:
    """
    Check if the ML API is available.
    
    Returns:
        True if API is available, False otherwise
    """
    try:
        response = requests.get(f"{DEFAULT_ML_API_URL}")
        return response.status_code == 200
    except:
        return False 