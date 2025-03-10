"""
ClinicalTrial model for storing information about clinical trials.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from typing import Optional, List
import enum

from ..database.init_db import Base


class TrialPhase(str, enum.Enum):
    """Enumeration of clinical trial phases."""
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
    NOT_APPLICABLE = "not_applicable"


class TrialStatus(str, enum.Enum):
    """Enumeration of clinical trial status."""
    RECRUITING = "recruiting"
    ACTIVE_NOT_RECRUITING = "active_not_recruiting"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class ClinicalTrial(Base):
    """SQLAlchemy model for clinical trial data."""
    __tablename__ = "clinical_trials"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Trial identification
    nct_id = Column(String(20), unique=True, index=True)  # ClinicalTrials.gov identifier
    title = Column(String(255))
    
    # Trial information
    description = Column(Text)
    phase = Column(String, default=TrialPhase.NOT_APPLICABLE.value)
    status = Column(String, default=TrialStatus.UNKNOWN.value)
    
    # Eligibility criteria
    conditions = Column(Text)  # Comma-separated list of conditions
    inclusion_criteria = Column(Text)
    exclusion_criteria = Column(Text)
    
    # Trial details
    start_date = Column(DateTime)
    completion_date = Column(DateTime)
    
    # Location information
    locations = Column(Text)  # JSON-encoded list of locations
    contact_name = Column(String(100))
    contact_email = Column(String(100))
    contact_phone = Column(String(20))
    
    # Relationship to specific treatment being tested
    treatment_id = Column(String, ForeignKey("treatments.id"), nullable=True)
    
    # Relationships
    treatment = relationship("Treatment", back_populates="clinical_trials")

    def __repr__(self):
        return f"<ClinicalTrial {self.title}, ID: {self.id}, NCT: {self.nct_id}>"


# Pydantic models for API request/response validation
class ClinicalTrialBase(BaseModel):
    """Base Pydantic model for clinical trial data."""
    nct_id: str
    title: str
    description: str
    phase: TrialPhase = TrialPhase.NOT_APPLICABLE
    status: TrialStatus = TrialStatus.UNKNOWN
    conditions: str
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    locations: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    treatment_id: Optional[str] = None


class ClinicalTrialCreate(ClinicalTrialBase):
    """Pydantic model for creating a new clinical trial."""
    pass


class ClinicalTrialUpdate(BaseModel):
    """Pydantic model for updating an existing clinical trial."""
    nct_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    phase: Optional[TrialPhase] = None
    status: Optional[TrialStatus] = None
    conditions: Optional[str] = None
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    locations: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    treatment_id: Optional[str] = None


class ClinicalTrialInDB(ClinicalTrialBase):
    """Pydantic model for clinical trial data from the database."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ClinicalTrialWithTreatment(ClinicalTrialInDB):
    """Pydantic model for clinical trial with related treatment details."""
    from .treatment import TreatmentInDB
    
    treatment: Optional[TreatmentInDB] = None 