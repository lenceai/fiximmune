"""
Treatment model for storing information about treatment options.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, Enum, ForeignKey
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from typing import Optional, List
import enum

from ..database.init_db import Base


class TreatmentType(str, enum.Enum):
    """Enumeration of treatment types."""
    MEDICATION = "medication"
    BIOLOGIC = "biologic"
    DIETARY = "dietary"
    PHYSICAL_THERAPY = "physical_therapy"
    SURGERY = "surgery"
    ALTERNATIVE = "alternative"
    LIFESTYLE = "lifestyle"
    EXPERIMENTAL = "experimental"
    OTHER = "other"


class TreatmentStatus(str, enum.Enum):
    """Enumeration of treatment status."""
    APPROVED = "approved"
    OFF_LABEL = "off_label"
    EXPERIMENTAL = "experimental"
    CLINICAL_TRIAL = "clinical_trial"


class Treatment(Base):
    """SQLAlchemy model for treatment data."""
    __tablename__ = "treatments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Basic treatment information
    name = Column(String(100), index=True)
    type = Column(String, default=TreatmentType.OTHER.value)
    description = Column(Text)
    
    # Treatment details
    status = Column(String, default=TreatmentStatus.APPROVED.value)
    dosage_info = Column(Text)
    administration_route = Column(String(50))
    
    # Effectiveness metrics
    avg_effectiveness = Column(Float, default=0.0)  # Scale 1-5
    
    # Side effects and contraindications
    side_effects = Column(Text)
    contraindications = Column(Text)
    
    # Additional information
    manufacturer = Column(String(100))
    approved_for = Column(Text)  # Comma-separated list of conditions
    
    # Flags
    is_experimental = Column(Boolean, default=False)
    
    # Relationships
    treatment_logs = relationship("TreatmentLog", back_populates="treatment")
    clinical_trials = relationship("ClinicalTrial", back_populates="treatment")

    def __repr__(self):
        return f"<Treatment {self.name}, ID: {self.id}>"


# Pydantic models for API request/response validation
class TreatmentBase(BaseModel):
    """Base Pydantic model for treatment data."""
    name: str
    type: TreatmentType = TreatmentType.OTHER
    description: str
    status: TreatmentStatus = TreatmentStatus.APPROVED
    dosage_info: Optional[str] = None
    administration_route: Optional[str] = None
    avg_effectiveness: float = 3.0
    side_effects: Optional[str] = None
    contraindications: Optional[str] = None
    manufacturer: Optional[str] = None
    approved_for: Optional[str] = None
    is_experimental: bool = False


class TreatmentCreate(TreatmentBase):
    """Pydantic model for creating a new treatment."""
    pass


class TreatmentUpdate(BaseModel):
    """Pydantic model for updating an existing treatment."""
    name: Optional[str] = None
    type: Optional[TreatmentType] = None
    description: Optional[str] = None
    status: Optional[TreatmentStatus] = None
    dosage_info: Optional[str] = None
    administration_route: Optional[str] = None
    avg_effectiveness: Optional[float] = None
    side_effects: Optional[str] = None
    contraindications: Optional[str] = None
    manufacturer: Optional[str] = None
    approved_for: Optional[str] = None
    is_experimental: Optional[bool] = None


class TreatmentInDB(TreatmentBase):
    """Pydantic model for treatment data from the database."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True 