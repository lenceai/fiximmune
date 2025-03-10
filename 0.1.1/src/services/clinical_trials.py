"""
Clinical trials service for fetching and processing clinical trial data.
"""

import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from ..models.patient import Patient
from ..models.clinical_trial import ClinicalTrial, TrialStatus, TrialPhase

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# API configuration
CLINICAL_TRIALS_API_KEY = os.getenv("CLINICAL_TRIALS_API_KEY")
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_clinical_trials(
    patient: Patient,
    db: Session,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetch relevant clinical trials for a patient.
    
    This function searches for clinical trials based on the patient's condition
    and returns the most relevant matches. It first checks the local database
    and then queries the ClinicalTrials.gov API for additional results.
    
    Args:
        patient: The patient object
        db: Database session
        limit: Maximum number of trials to return
        
    Returns:
        List of clinical trials relevant to the patient
    """
    logger.info(f"Fetching clinical trials for patient {patient.id} with condition {patient.disease_type}")
    
    # First, check local database for relevant trials
    local_trials = get_local_clinical_trials(patient, db)
    
    # Then, fetch from external API if we need more results
    if len(local_trials) < limit:
        external_trials = get_external_clinical_trials(patient, limit - len(local_trials))
        
        # Save new trials to the database
        for trial in external_trials:
            save_clinical_trial(trial, db)
        
        # Combine results
        all_trials = local_trials + external_trials
    else:
        all_trials = local_trials
    
    # Limit results
    return all_trials[:limit]


def get_local_clinical_trials(patient: Patient, db: Session) -> List[Dict[str, Any]]:
    """
    Get relevant clinical trials from the local database.
    
    Args:
        patient: The patient object
        db: Database session
    
    Returns:
        List of relevant clinical trials from the database
    """
    # Query clinical trials related to the patient's condition
    trials = db.query(ClinicalTrial).filter(
        ClinicalTrial.conditions.contains(patient.disease_type),
        ClinicalTrial.status == TrialStatus.RECRUITING.value
    ).all()
    
    return [
        {
            "id": trial.id,
            "nct_id": trial.nct_id,
            "title": trial.title,
            "phase": trial.phase,
            "status": trial.status,
            "description": trial.description,
            "conditions": trial.conditions,
            "locations": trial.locations,
            "start_date": trial.start_date.isoformat() if trial.start_date else None,
            "completion_date": trial.completion_date.isoformat() if trial.completion_date else None,
            "source": "database"
        }
        for trial in trials
    ]


def get_external_clinical_trials(patient: Patient, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch clinical trials from ClinicalTrials.gov API.
    
    In a real implementation, this would use the actual API.
    This function provides a simulation of what that might look like.
    
    Args:
        patient: The patient object
        limit: Maximum number of trials to return
    
    Returns:
        List of clinical trials from the external API
    """
    logger.info(f"Fetching external clinical trials for condition: {patient.disease_type}")
    
    # In a real implementation, you would call the actual API
    # For this MVP, we simulate the API response
    try:
        # Simulate API call with mock data
        # In production, this would be a real API call:
        # response = requests.get(
        #     CLINICAL_TRIALS_API_URL,
        #     params={
        #         "condition": patient.disease_type,
        #         "status": "recruiting",
        #         "limit": limit
        #     },
        #     headers={"Authorization": f"Bearer {CLINICAL_TRIALS_API_KEY}"}
        # )
        # response.raise_for_status()
        # trials = response.json()["studies"]
        
        # Mock response for MVP
        trials = [
            {
                "nct_id": f"NCT0{i}",
                "title": f"Clinical Trial for {patient.disease_type.replace('_', ' ').title()} - Study {i}",
                "phase": TrialPhase.PHASE_2.value if i % 3 != 0 else TrialPhase.PHASE_3.value,
                "status": TrialStatus.RECRUITING.value,
                "description": f"A study investigating new treatments for {patient.disease_type.replace('_', ' ')}.",
                "conditions": patient.disease_type,
                "locations": json.dumps([{"name": "Medical Center", "city": "New York", "state": "NY", "country": "USA"}]),
                "start_date": datetime.now().isoformat(),
                "completion_date": datetime(2025, 12, 31).isoformat(),
                "contact_name": "Trial Coordinator",
                "contact_email": "contact@trial.org",
                "contact_phone": "123-456-7890",
                "inclusion_criteria": "Adults aged 18-65 with confirmed diagnosis",
                "exclusion_criteria": "Pregnant women, severe comorbidities"
            }
            for i in range(1, limit + 1)
        ]
        
        # Add source field to indicate these came from the external API
        for trial in trials:
            trial["source"] = "clinicaltrials.gov"
        
        return trials
    
    except Exception as e:
        logger.error(f"Error fetching clinical trials: {e}")
        return []


def save_clinical_trial(trial_data: Dict[str, Any], db: Session) -> Optional[ClinicalTrial]:
    """
    Save a clinical trial to the database.
    
    Args:
        trial_data: Clinical trial data
        db: Database session
    
    Returns:
        The saved clinical trial object or None if failed
    """
    try:
        # Check if trial already exists
        existing_trial = db.query(ClinicalTrial).filter(
            ClinicalTrial.nct_id == trial_data["nct_id"]
        ).first()
        
        if existing_trial:
            logger.info(f"Clinical trial {trial_data['nct_id']} already exists in database")
            return existing_trial
        
        # Create new clinical trial
        # Remove source field as it's not part of the model
        if "source" in trial_data:
            del trial_data["source"]
        
        # Convert string dates to datetime objects
        for date_field in ["start_date", "completion_date"]:
            if date_field in trial_data and isinstance(trial_data[date_field], str):
                try:
                    trial_data[date_field] = datetime.fromisoformat(trial_data[date_field])
                except ValueError:
                    trial_data[date_field] = None
        
        new_trial = ClinicalTrial(**trial_data)
        db.add(new_trial)
        db.commit()
        db.refresh(new_trial)
        
        logger.info(f"Saved new clinical trial: {new_trial.nct_id}")
        return new_trial
    
    except Exception as e:
        logger.error(f"Error saving clinical trial: {e}")
        db.rollback()
        return None 