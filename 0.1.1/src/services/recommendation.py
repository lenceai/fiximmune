"""
Treatment recommendation service for generating personalized treatment suggestions.
"""

import os
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from dotenv import load_dotenv

from ..models.patient import Patient, AutoimmuneDisease
from ..models.treatment import Treatment, TreatmentType
from ..models.treatment_log import TreatmentLog

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def get_treatment_recommendations(
    patient: Patient,
    include_experimental: bool = False,
    db: Session = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Generate personalized treatment recommendations for a patient.
    
    This function analyzes patient data and historical treatment effectiveness
    to provide a ranked list of recommended treatments.
    
    Args:
        patient: The patient object
        include_experimental: Whether to include experimental treatments
        db: Database session
        limit: Maximum number of recommendations to return
        
    Returns:
        Dictionary containing recommended treatments and supporting information
    """
    logger.info(f"Generating treatment recommendations for patient {patient.id}")
    
    # Get disease-specific treatments
    disease_specific_treatments = get_disease_specific_treatments(
        patient.disease_type, 
        include_experimental, 
        db
    )
    
    # Get treatments that have been effective for similar patients
    similar_patient_treatments = get_treatments_for_similar_patients(
        patient, 
        include_experimental, 
        db
    )
    
    # Get general effectiveness-based recommendations
    effectiveness_based_treatments = get_effectiveness_based_treatments(
        include_experimental, 
        db
    )
    
    # Combine and rank all recommendations
    all_recommendations = rank_and_combine_recommendations(
        disease_specific_treatments,
        similar_patient_treatments,
        effectiveness_based_treatments,
        patient,
        limit
    )
    
    # Format response
    response = {
        "patient_id": patient.id,
        "disease_type": patient.disease_type,
        "recommendations": all_recommendations,
        "include_experimental": include_experimental
    }
    
    return response


def get_disease_specific_treatments(
    disease_type: str,
    include_experimental: bool,
    db: Session
) -> List[Dict[str, Any]]:
    """
    Get treatments that are specific to a particular autoimmune disease.
    
    Args:
        disease_type: The type of autoimmune disease
        include_experimental: Whether to include experimental treatments
        db: Database session
    
    Returns:
        List of treatments specific to the disease
    """
    query = db.query(Treatment).filter(
        Treatment.approved_for.contains(disease_type)
    ).order_by(desc(Treatment.avg_effectiveness))
    
    if not include_experimental:
        query = query.filter(Treatment.is_experimental == False)
    
    treatments = query.all()
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "type": t.type,
            "avg_effectiveness": t.avg_effectiveness,
            "is_experimental": t.is_experimental,
            "source": "disease_specific",
            "confidence_score": 0.9 if not t.is_experimental else 0.7,
        }
        for t in treatments
    ]


def get_treatments_for_similar_patients(
    patient: Patient,
    include_experimental: bool,
    db: Session
) -> List[Dict[str, Any]]:
    """
    Get treatments that have been effective for similar patients.
    
    Args:
        patient: The patient object
        include_experimental: Whether to include experimental treatments
        db: Database session
    
    Returns:
        List of treatments effective for similar patients
    """
    # Find patients with the same disease type
    similar_patients = db.query(Patient).filter(
        Patient.disease_type == patient.disease_type,
        Patient.id != patient.id
    ).all()
    
    similar_patient_ids = [p.id for p in similar_patients]
    
    if not similar_patient_ids:
        return []
    
    # Find effective treatments for these patients
    effective_treatment_logs = db.query(TreatmentLog).filter(
        TreatmentLog.patient_id.in_(similar_patient_ids),
        TreatmentLog.effectiveness_rating >= 3  # At least moderate improvement
    ).all()
    
    if not effective_treatment_logs:
        return []
    
    # Get unique treatment IDs
    treatment_ids = list(set([log.treatment_id for log in effective_treatment_logs]))
    
    # Query the treatments
    query = db.query(Treatment).filter(Treatment.id.in_(treatment_ids))
    
    if not include_experimental:
        query = query.filter(Treatment.is_experimental == False)
    
    treatments = query.all()
    
    # Create a mapping of treatments to their average effectiveness
    treatment_effectiveness = {}
    for log in effective_treatment_logs:
        if log.treatment_id not in treatment_effectiveness:
            treatment_effectiveness[log.treatment_id] = {"total": 0, "count": 0}
        
        treatment_effectiveness[log.treatment_id]["total"] += log.effectiveness_rating
        treatment_effectiveness[log.treatment_id]["count"] += 1
    
    # Calculate average effectiveness
    for treatment_id in treatment_effectiveness:
        data = treatment_effectiveness[treatment_id]
        treatment_effectiveness[treatment_id] = data["total"] / data["count"]
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "type": t.type,
            "avg_effectiveness": treatment_effectiveness.get(t.id, t.avg_effectiveness),
            "is_experimental": t.is_experimental,
            "source": "similar_patients",
            "confidence_score": 0.85 if not t.is_experimental else 0.65,
        }
        for t in treatments
    ]


def get_effectiveness_based_treatments(
    include_experimental: bool,
    db: Session
) -> List[Dict[str, Any]]:
    """
    Get treatments based on overall effectiveness ratings.
    
    Args:
        include_experimental: Whether to include experimental treatments
        db: Database session
    
    Returns:
        List of treatments ranked by overall effectiveness
    """
    query = db.query(Treatment).order_by(desc(Treatment.avg_effectiveness))
    
    if not include_experimental:
        query = query.filter(Treatment.is_experimental == False)
    
    treatments = query.limit(10).all()
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "type": t.type,
            "avg_effectiveness": t.avg_effectiveness,
            "is_experimental": t.is_experimental,
            "source": "overall_effectiveness",
            "confidence_score": 0.7 if not t.is_experimental else 0.5,
        }
        for t in treatments
    ]


def rank_and_combine_recommendations(
    disease_specific_treatments: List[Dict[str, Any]],
    similar_patient_treatments: List[Dict[str, Any]],
    effectiveness_based_treatments: List[Dict[str, Any]],
    patient: Patient,
    limit: int
) -> List[Dict[str, Any]]:
    """
    Combine and rank all treatment recommendations.
    
    Args:
        disease_specific_treatments: Treatments specific to the disease
        similar_patient_treatments: Treatments effective for similar patients
        effectiveness_based_treatments: Treatments with high overall effectiveness
        patient: The patient object
        limit: Maximum number of recommendations to return
    
    Returns:
        Ranked list of combined treatment recommendations
    """
    # Combine all treatments
    all_treatments = disease_specific_treatments + similar_patient_treatments + effectiveness_based_treatments
    
    # Remove duplicates, prioritizing treatments from more specific sources
    treatment_map = {}
    for treatment in all_treatments:
        treatment_id = treatment["id"]
        
        if treatment_id not in treatment_map:
            treatment_map[treatment_id] = treatment
        else:
            # If treatment already exists, keep the one with the higher confidence score
            if treatment["confidence_score"] > treatment_map[treatment_id]["confidence_score"]:
                treatment_map[treatment_id] = treatment
    
    # Convert back to list
    combined_treatments = list(treatment_map.values())
    
    # Sort by confidence score and effectiveness
    ranked_treatments = sorted(
        combined_treatments, 
        key=lambda x: (x["confidence_score"], x["avg_effectiveness"]), 
        reverse=True
    )
    
    # Return limited number of recommendations
    return ranked_treatments[:limit] 