"""
Patient model for storing patient information.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Enum
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from typing import Optional, List
import enum

from ..database.init_db import Base


class AutoimmuneDisease(str, enum.Enum):
    """Enumeration of autoimmune diseases."""
    RHEUMATOID_ARTHRITIS = "rheumatoid_arthritis"
    LUPUS = "lupus"
    MULTIPLE_SCLEROSIS = "multiple_sclerosis"
    TYPE_1_DIABETES = "type_1_diabetes"
    PSORIASIS = "psoriasis"
    CROHNS_DISEASE = "crohns_disease"
    ULCERATIVE_COLITIS = "ulcerative_colitis"
    CELIAC_DISEASE = "celiac_disease"
    GRAVES_DISEASE = "graves_disease"
    HASHIMOTOS_THYROIDITIS = "hashimotos_thyroiditis"
    OTHER = "other"


class SymptomSeverity(int, enum.Enum):
    """Enumeration of symptom severity levels."""
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    VERY_SEVERE = 4
    DEBILITATING = 5


class Patient(Base):
    """SQLAlchemy model for patient data."""
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Basic demographic information
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(20))
    weight = Column(Float)  # in kg
    height = Column(Float)  # in cm
    
    # Disease information
    disease_type = Column(String, default=AutoimmuneDisease.OTHER.value)
    diagnosis_date = Column(DateTime)
    symptom_severity = Column(Integer, default=SymptomSeverity.MODERATE.value)
    
    # Additional medical information
    medical_history = Column(Text)
    allergies = Column(Text)
    
    # Relationships
    treatment_logs = relationship("TreatmentLog", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient {self.name}, ID: {self.id}>"


# Pydantic models for API request/response validation
class PatientBase(BaseModel):
    """Base Pydantic model for patient data."""
    name: str
    age: int
    gender: str
    weight: Optional[float] = None
    height: Optional[float] = None
    disease_type: AutoimmuneDisease = AutoimmuneDisease.OTHER
    diagnosis_date: Optional[datetime] = None
    symptom_severity: SymptomSeverity = SymptomSeverity.MODERATE
    medical_history: Optional[str] = None
    allergies: Optional[str] = None


class PatientCreate(PatientBase):
    """Pydantic model for creating a new patient."""
    pass


class PatientUpdate(BaseModel):
    """Pydantic model for updating an existing patient."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    disease_type: Optional[AutoimmuneDisease] = None
    diagnosis_date: Optional[datetime] = None
    symptom_severity: Optional[SymptomSeverity] = None
    medical_history: Optional[str] = None
    allergies: Optional[str] = None


class PatientInDB(PatientBase):
    """Pydantic model for patient data from the database."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True 