"""
Tests for sample data initialization in FixImmune.
"""

import pytest
import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.init_db import Base
from src.utils.sample_data import (
    load_treatments,
    load_clinical_trials,
    create_sample_patients,
    create_sample_treatment_logs,
    initialize_sample_data
)

# Create test database
@pytest.fixture
def test_db():
    """Create a temporary SQLite database for testing."""
    # Create a temporary file for the test database
    _, temp_db_path = tempfile.mkstemp(suffix=".db")
    db_url = f"sqlite:///{temp_db_path}"
    
    # Create engine and session
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Get a test session
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        # Clean up the temporary file
        os.unlink(temp_db_path)

def test_load_treatments(test_db):
    """Test loading treatments into the database."""
    treatments = load_treatments(test_db)
    assert len(treatments) > 0
    
    # Check if we have different types of treatments
    treatment_types = set(t.type for t in treatments)
    assert len(treatment_types) > 1
    
    # Check if we have some experimental treatments
    experimental_treatments = [t for t in treatments if t.is_experimental]
    assert len(experimental_treatments) > 0

def test_load_clinical_trials(test_db):
    """Test loading clinical trials into the database."""
    # First need to load treatments as they're referenced by trials
    load_treatments(test_db)
    
    trials = load_clinical_trials(test_db)
    assert len(trials) > 0
    
    # Check if we have different phases of trials
    trial_phases = set(t.phase for t in trials)
    assert len(trial_phases) > 1

def test_create_sample_patients(test_db):
    """Test creating sample patients in the database."""
    patients = create_sample_patients(test_db)
    assert len(patients) > 0
    
    # Check if we have patients with different autoimmune diseases
    disease_types = set(p.disease_type for p in patients)
    assert len(disease_types) > 1

def test_create_sample_treatment_logs(test_db):
    """Test creating sample treatment logs in the database."""
    # First need to load treatments and create patients
    load_treatments(test_db)
    create_sample_patients(test_db)
    
    logs = create_sample_treatment_logs(test_db)
    assert len(logs) > 0
    
    # Check if we have different effectiveness ratings
    effectiveness_ratings = set(log.effectiveness_rating for log in logs)
    assert len(effectiveness_ratings) > 1

def test_initialize_sample_data(test_db):
    """Test the complete sample data initialization workflow."""
    # Call the main initialization function
    initialize_sample_data(test_db)
    
    # Check that we have data in all tables
    from src.models.treatment import Treatment
    from src.models.clinical_trial import ClinicalTrial
    from src.models.patient import Patient
    from src.models.treatment_log import TreatmentLog
    
    treatments_count = test_db.query(Treatment).count()
    trials_count = test_db.query(ClinicalTrial).count()
    patients_count = test_db.query(Patient).count()
    logs_count = test_db.query(TreatmentLog).count()
    
    assert treatments_count > 0, "No treatments were created"
    assert trials_count > 0, "No clinical trials were created"
    assert patients_count > 0, "No patients were created"
    assert logs_count > 0, "No treatment logs were created" 