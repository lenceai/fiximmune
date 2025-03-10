"""
API routes for FixImmune application.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from ..database.init_db import get_db
from ..models.patient import Patient, PatientCreate, PatientUpdate, PatientInDB
from ..models.treatment import Treatment, TreatmentCreate, TreatmentUpdate, TreatmentInDB
from ..models.treatment_log import TreatmentLog, TreatmentLogCreate, TreatmentLogUpdate, TreatmentLogInDB, TreatmentLogWithDetails
from ..models.clinical_trial import ClinicalTrial, ClinicalTrialCreate, ClinicalTrialUpdate, ClinicalTrialInDB
from ..services.recommendation import get_treatment_recommendations
from ..services.clinical_trials import fetch_clinical_trials

# Import ML API integration if available
try:
    from ..ml.api.integration import (
        get_treatment_recommendations_for_patient,
        check_ml_api_status,
        MLApiClient
    )
    ml_api_available = True
except ImportError:
    ml_api_available = False

# Create router
router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)

# Patient routes
@router.post("/patients/", response_model=PatientInDB)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient."""
    db_patient = Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@router.get("/patients/", response_model=List[PatientInDB])
def get_all_patients(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Get all patients."""
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return patients

@router.get("/patients/{patient_id}", response_model=PatientInDB)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Get a specific patient by ID."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/patients/{patient_id}", response_model=PatientInDB)
def update_patient(
    patient_id: str, 
    patient_update: PatientUpdate, 
    db: Session = Depends(get_db)
):
    """Update a patient."""
    db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if db_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Update patient attributes
    update_data = patient_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_patient, key, value)
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@router.delete("/patients/{patient_id}")
def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    """Delete a patient."""
    db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if db_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    db.delete(db_patient)
    db.commit()
    return {"ok": True}

# Treatment routes
@router.post("/treatments/", response_model=TreatmentInDB)
def create_treatment(treatment: TreatmentCreate, db: Session = Depends(get_db)):
    """Create a new treatment."""
    db_treatment = Treatment(**treatment.dict())
    db.add(db_treatment)
    db.commit()
    db.refresh(db_treatment)
    return db_treatment

@router.get("/treatments/", response_model=List[TreatmentInDB])
def get_all_treatments(
    skip: int = 0, 
    limit: int = 100, 
    is_experimental: Optional[bool] = None,
    type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all treatments with optional filtering."""
    query = db.query(Treatment)
    
    if is_experimental is not None:
        query = query.filter(Treatment.is_experimental == is_experimental)
    
    if type is not None:
        query = query.filter(Treatment.type == type)
    
    treatments = query.offset(skip).limit(limit).all()
    return treatments

@router.get("/treatments/{treatment_id}", response_model=TreatmentInDB)
def get_treatment(treatment_id: str, db: Session = Depends(get_db)):
    """Get a specific treatment by ID."""
    treatment = db.query(Treatment).filter(Treatment.id == treatment_id).first()
    if treatment is None:
        raise HTTPException(status_code=404, detail="Treatment not found")
    return treatment

@router.put("/treatments/{treatment_id}", response_model=TreatmentInDB)
def update_treatment(
    treatment_id: str, 
    treatment_update: TreatmentUpdate, 
    db: Session = Depends(get_db)
):
    """Update a treatment."""
    db_treatment = db.query(Treatment).filter(Treatment.id == treatment_id).first()
    if db_treatment is None:
        raise HTTPException(status_code=404, detail="Treatment not found")
    
    # Update treatment attributes
    update_data = treatment_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_treatment, key, value)
    
    db.add(db_treatment)
    db.commit()
    db.refresh(db_treatment)
    return db_treatment

@router.delete("/treatments/{treatment_id}")
def delete_treatment(treatment_id: str, db: Session = Depends(get_db)):
    """Delete a treatment."""
    db_treatment = db.query(Treatment).filter(Treatment.id == treatment_id).first()
    if db_treatment is None:
        raise HTTPException(status_code=404, detail="Treatment not found")
    
    db.delete(db_treatment)
    db.commit()
    return {"ok": True}

# Treatment Log routes
@router.post("/treatment-logs/", response_model=TreatmentLogInDB)
def create_treatment_log(treatment_log: TreatmentLogCreate, db: Session = Depends(get_db)):
    """Create a new treatment log entry."""
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == treatment_log.patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Verify treatment exists
    treatment = db.query(Treatment).filter(Treatment.id == treatment_log.treatment_id).first()
    if treatment is None:
        raise HTTPException(status_code=404, detail="Treatment not found")
    
    db_treatment_log = TreatmentLog(**treatment_log.dict())
    db.add(db_treatment_log)
    db.commit()
    db.refresh(db_treatment_log)
    return db_treatment_log

@router.get("/treatment-logs/", response_model=List[TreatmentLogInDB])
def get_all_treatment_logs(
    skip: int = 0, 
    limit: int = 100, 
    patient_id: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get all treatment logs with optional filtering."""
    query = db.query(TreatmentLog)
    
    if patient_id is not None:
        query = query.filter(TreatmentLog.patient_id == patient_id)
    
    if is_active is not None:
        query = query.filter(TreatmentLog.is_active == is_active)
    
    treatment_logs = query.offset(skip).limit(limit).all()
    return treatment_logs

@router.get("/treatment-logs/{log_id}", response_model=TreatmentLogWithDetails)
def get_treatment_log(log_id: str, db: Session = Depends(get_db)):
    """Get a specific treatment log by ID with detailed information."""
    treatment_log = db.query(TreatmentLog).filter(TreatmentLog.id == log_id).first()
    if treatment_log is None:
        raise HTTPException(status_code=404, detail="Treatment log not found")
    
    # Eager load treatment and patient
    treatment = db.query(Treatment).filter(Treatment.id == treatment_log.treatment_id).first()
    patient = db.query(Patient).filter(Patient.id == treatment_log.patient_id).first()
    
    # Create response object
    response = TreatmentLogWithDetails.from_orm(treatment_log)
    response.treatment = treatment
    response.patient = patient
    
    return response

@router.put("/treatment-logs/{log_id}", response_model=TreatmentLogInDB)
def update_treatment_log(
    log_id: str, 
    treatment_log_update: TreatmentLogUpdate, 
    db: Session = Depends(get_db)
):
    """Update a treatment log."""
    db_treatment_log = db.query(TreatmentLog).filter(TreatmentLog.id == log_id).first()
    if db_treatment_log is None:
        raise HTTPException(status_code=404, detail="Treatment log not found")
    
    # Update treatment log attributes
    update_data = treatment_log_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_treatment_log, key, value)
    
    db.add(db_treatment_log)
    db.commit()
    db.refresh(db_treatment_log)
    return db_treatment_log

@router.delete("/treatment-logs/{log_id}")
def delete_treatment_log(log_id: str, db: Session = Depends(get_db)):
    """Delete a treatment log."""
    db_treatment_log = db.query(TreatmentLog).filter(TreatmentLog.id == log_id).first()
    if db_treatment_log is None:
        raise HTTPException(status_code=404, detail="Treatment log not found")
    
    db.delete(db_treatment_log)
    db.commit()
    return {"ok": True}

# Clinical Trial routes
@router.post("/clinical-trials/", response_model=ClinicalTrialInDB)
def create_clinical_trial(clinical_trial: ClinicalTrialCreate, db: Session = Depends(get_db)):
    """Create a new clinical trial."""
    db_clinical_trial = ClinicalTrial(**clinical_trial.dict())
    db.add(db_clinical_trial)
    db.commit()
    db.refresh(db_clinical_trial)
    return db_clinical_trial

@router.get("/clinical-trials/", response_model=List[ClinicalTrialInDB])
def get_all_clinical_trials(
    skip: int = 0, 
    limit: int = 100, 
    condition: Optional[str] = None,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all clinical trials with optional filtering."""
    query = db.query(ClinicalTrial)
    
    if condition is not None:
        query = query.filter(ClinicalTrial.conditions.contains(condition))
    
    if status is not None:
        query = query.filter(ClinicalTrial.status == status)
    
    if phase is not None:
        query = query.filter(ClinicalTrial.phase == phase)
    
    clinical_trials = query.offset(skip).limit(limit).all()
    return clinical_trials

@router.get("/clinical-trials/{trial_id}", response_model=ClinicalTrialInDB)
def get_clinical_trial(trial_id: str, db: Session = Depends(get_db)):
    """Get a specific clinical trial by ID."""
    clinical_trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if clinical_trial is None:
        raise HTTPException(status_code=404, detail="Clinical trial not found")
    return clinical_trial

@router.put("/clinical-trials/{trial_id}", response_model=ClinicalTrialInDB)
def update_clinical_trial(
    trial_id: str, 
    clinical_trial_update: ClinicalTrialUpdate, 
    db: Session = Depends(get_db)
):
    """Update a clinical trial."""
    db_clinical_trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if db_clinical_trial is None:
        raise HTTPException(status_code=404, detail="Clinical trial not found")
    
    # Update clinical trial attributes
    update_data = clinical_trial_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_clinical_trial, key, value)
    
    db.add(db_clinical_trial)
    db.commit()
    db.refresh(db_clinical_trial)
    return db_clinical_trial

@router.delete("/clinical-trials/{trial_id}")
def delete_clinical_trial(trial_id: str, db: Session = Depends(get_db)):
    """Delete a clinical trial."""
    db_clinical_trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if db_clinical_trial is None:
        raise HTTPException(status_code=404, detail="Clinical trial not found")
    
    db.delete(db_clinical_trial)
    db.commit()
    return {"ok": True}

# Treatment Recommendation routes
@router.get("/recommendations/{patient_id}")
def get_recommendations(
    patient_id: str, 
    include_experimental: bool = False,
    include_clinical_trials: bool = False,
    use_ml_model: bool = False,
    model_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get personalized treatment recommendations for a patient."""
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Use ML model for recommendations if requested and available
    if use_ml_model and ml_api_available:
        try:
            # Check if ML API is available
            if not check_ml_api_status():
                logger.warning("ML API is not available, falling back to rule-based recommendations")
                use_ml_model = False
            else:
                # Convert patient model to dictionary
                patient_dict = {
                    "id": patient.id,
                    "age": patient.age,
                    "gender": patient.gender,
                    "weight": patient.weight,
                    "height": patient.height,
                    "disease_type": patient.disease_type,
                    "symptom_severity": patient.symptom_severity,
                    "diagnosis_date": patient.diagnosis_date.isoformat() if patient.diagnosis_date else None,
                    "medical_history": patient.medical_history,
                    "allergies": patient.allergies
                }
                
                # Get treatment logs for this patient to add as previous treatments
                treatment_logs = db.query(TreatmentLog).filter(TreatmentLog.patient_id == patient_id).all()
                if treatment_logs:
                    # Get treatment IDs from logs
                    treatment_ids = [log.treatment_id for log in treatment_logs]
                    patient_dict["previous_treatments"] = treatment_ids
                
                # Get ML-based recommendations
                ml_recommendations = get_treatment_recommendations_for_patient(
                    patient_id=patient_id,
                    patient_data=patient_dict,
                    include_experimental=include_experimental,
                    model_id=model_id
                )
                
                # Check for errors
                if "error" in ml_recommendations:
                    logger.error(f"Error from ML API: {ml_recommendations['error']}")
                    # Fall back to rule-based recommendations
                    use_ml_model = False
                else:
                    # Process ML recommendations
                    recommendations = {
                        "patient_id": patient_id,
                        "disease_type": patient.disease_type,
                        "recommendations": [],
                        "include_experimental": include_experimental,
                        "ml_model_id": ml_recommendations.get("model_id")
                    }
                    
                    # Convert ML predictions to consistent format
                    for pred in ml_recommendations.get("predictions", []):
                        # Try to get treatment details from database
                        treatment = db.query(Treatment).filter(Treatment.id == pred.get("treatment_id")).first()
                        
                        if treatment:
                            # Use treatment from database
                            recommendation = {
                                "id": treatment.id,
                                "name": treatment.name,
                                "type": treatment.type,
                                "avg_effectiveness": pred.get("predicted_effectiveness", treatment.avg_effectiveness),
                                "is_experimental": treatment.is_experimental,
                                "confidence": pred.get("confidence", 0.8),
                                "source": "ml_model"
                            }
                        else:
                            # Use prediction data directly
                            recommendation = {
                                "id": pred.get("treatment_id", "unknown"),
                                "name": pred.get("treatment_name", "Unknown Treatment"),
                                "type": "unknown",
                                "avg_effectiveness": pred.get("predicted_effectiveness", 3.0),
                                "is_experimental": False,
                                "confidence": pred.get("confidence", 0.8),
                                "source": "ml_model"
                            }
                        
                        recommendations["recommendations"].append(recommendation)
        except Exception as e:
            logger.error(f"Error using ML model: {e}")
            # Fall back to rule-based recommendations
            use_ml_model = False
    
    # If ML model not requested or not available, use rule-based approach
    if not use_ml_model or not ml_api_available:
        # Get rule-based treatment recommendations
        recommendations = get_treatment_recommendations(
            patient, 
            include_experimental=include_experimental,
            db=db
        )
    
    # Get clinical trial recommendations if requested
    if include_clinical_trials:
        clinical_trials = fetch_clinical_trials(patient, db=db)
        recommendations["clinical_trials"] = clinical_trials
    
    return recommendations

# ML API integration routes (if available)
if ml_api_available:
    @router.get("/ml/status")
    def check_ml_api():
        """Check if the ML API is available."""
        is_available = check_ml_api_status()
        return {"available": is_available}
    
    @router.get("/ml/models")
    def list_ml_models():
        """List available ML models."""
        try:
            client = MLApiClient()
            models = client.list_models()
            return {"models": models}
        except Exception as e:
            logger.error(f"Error listing ML models: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing ML models: {str(e)}")
    
    @router.get("/ml/models/{model_id}")
    def get_ml_model_info(model_id: str):
        """Get information about a specific ML model."""
        try:
            client = MLApiClient()
            model_info = client.get_model_info(model_id)
            
            if "error" in model_info:
                raise HTTPException(status_code=404, detail=f"Model not found: {model_info['error']}")
            
            return model_info
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting ML model info: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting ML model info: {str(e)}") 