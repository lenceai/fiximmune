"""
TreatmentLog model for tracking patient treatments and outcomes.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from typing import Optional, List
import enum

from ..database.init_db import Base


class EffectivenessRating(int, enum.Enum):
    """Enumeration of treatment effectiveness ratings."""
    NO_IMPROVEMENT = 1
    SLIGHT_IMPROVEMENT = 2
    MODERATE_IMPROVEMENT = 3
    SIGNIFICANT_IMPROVEMENT = 4
    CURED = 5


class TreatmentLog(Base):
    """SQLAlchemy model for treatment log data."""
    __tablename__ = "treatment_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    patient_id = Column(String, ForeignKey("patients.id", ondelete="CASCADE"))
    treatment_id = Column(String, ForeignKey("treatments.id"))
    
    # Treatment period
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)  # Null if ongoing
    
    # Treatment details
    dosage = Column(String(100))
    frequency = Column(String(100))
    
    # Effectiveness and outcomes
    effectiveness_rating = Column(Integer, default=EffectivenessRating.MODERATE_IMPROVEMENT.value)
    side_effects_experienced = Column(Text)
    notes = Column(Text)
    
    # Status flags
    is_active = Column(Boolean, default=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="treatment_logs")
    treatment = relationship("Treatment", back_populates="treatment_logs")

    def __repr__(self):
        return f"<TreatmentLog {self.id}, Patient: {self.patient_id}, Treatment: {self.treatment_id}>"


# Pydantic models for API request/response validation
class TreatmentLogBase(BaseModel):
    """Base Pydantic model for treatment log data."""
    patient_id: str
    treatment_id: str
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    effectiveness_rating: EffectivenessRating = EffectivenessRating.MODERATE_IMPROVEMENT
    side_effects_experienced: Optional[str] = None
    notes: Optional[str] = None
    is_active: bool = True


class TreatmentLogCreate(TreatmentLogBase):
    """Pydantic model for creating a new treatment log."""
    pass


class TreatmentLogUpdate(BaseModel):
    """Pydantic model for updating an existing treatment log."""
    end_date: Optional[datetime] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    effectiveness_rating: Optional[EffectivenessRating] = None
    side_effects_experienced: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class TreatmentLogInDB(TreatmentLogBase):
    """Pydantic model for treatment log data from the database."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class TreatmentLogWithDetails(TreatmentLogInDB):
    """Pydantic model for treatment log with related treatment and patient details."""
    from .treatment import TreatmentInDB
    from .patient import PatientInDB
    
    treatment: Optional[TreatmentInDB] = None
    patient: Optional[PatientInDB] = None 