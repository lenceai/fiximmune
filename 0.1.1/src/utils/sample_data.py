"""
Sample data loader for FixImmune.

This module provides functions to initialize the database with sample data
for testing and demonstration purposes.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from ..models.patient import Patient, AutoimmuneDisease, SymptomSeverity
from ..models.treatment import Treatment, TreatmentType, TreatmentStatus
from ..models.treatment_log import TreatmentLog, EffectivenessRating
from ..models.clinical_trial import ClinicalTrial, TrialPhase, TrialStatus

# Configure logging
logger = logging.getLogger(__name__)

def load_sample_data(db: Session) -> None:
    """
    Load all sample data into the database.
    
    Args:
        db: Database session
    """
    logger.info("Loading sample data...")
    
    # Load data in correct order to maintain relationships
    load_treatments(db)
    load_clinical_trials(db)
    
    logger.info("Sample data loaded successfully.")

def load_treatments(db: Session) -> List[Treatment]:
    """
    Load sample treatments into the database.
    
    Args:
        db: Database session
        
    Returns:
        List of created treatment objects
    """
    logger.info("Loading sample treatments...")
    
    # Define sample treatments
    treatments_data = [
        # Rheumatoid Arthritis Treatments
        {
            "name": "Methotrexate",
            "type": TreatmentType.MEDICATION.value,
            "description": "A disease-modifying anti-rheumatic drug (DMARD) that can reduce joint damage and disability.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "7.5-25 mg once weekly",
            "administration_route": "Oral or injection",
            "avg_effectiveness": 3.8,
            "side_effects": "Nausea, mouth sores, fatigue, liver function abnormalities",
            "contraindications": "Pregnancy, liver disease, alcohol consumption",
            "manufacturer": "Various",
            "approved_for": "rheumatoid_arthritis, psoriasis",
            "is_experimental": False
        },
        {
            "name": "Humira (Adalimumab)",
            "type": TreatmentType.BIOLOGIC.value,
            "description": "A TNF inhibitor that reduces inflammation and stops disease progression.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "40 mg every other week",
            "administration_route": "Subcutaneous injection",
            "avg_effectiveness": 4.2,
            "side_effects": "Injection site reactions, increased risk of infections",
            "contraindications": "Active tuberculosis, serious infections",
            "manufacturer": "AbbVie",
            "approved_for": "rheumatoid_arthritis, psoriasis, crohns_disease, ulcerative_colitis",
            "is_experimental": False
        },
        {
            "name": "Prednisone",
            "type": TreatmentType.MEDICATION.value,
            "description": "A corticosteroid that reduces inflammation and suppresses the immune system.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "5-60 mg daily, depending on severity",
            "administration_route": "Oral",
            "avg_effectiveness": 4.0,
            "side_effects": "Weight gain, fluid retention, bone loss, increased blood sugar",
            "contraindications": "Fungal infections, certain viral infections",
            "manufacturer": "Various",
            "approved_for": "rheumatoid_arthritis, lupus, multiple_sclerosis, crohns_disease, ulcerative_colitis",
            "is_experimental": False
        },
        
        # Multiple Sclerosis Treatments
        {
            "name": "Ocrevus (Ocrelizumab)",
            "type": TreatmentType.BIOLOGIC.value,
            "description": "A monoclonal antibody that targets CD20-positive B cells, a type of immune cell thought to contribute to nerve damage in MS.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "600 mg every 6 months",
            "administration_route": "Intravenous infusion",
            "avg_effectiveness": 4.3,
            "side_effects": "Infusion reactions, increased risk of infections",
            "contraindications": "Active hepatitis B infection",
            "manufacturer": "Genentech (Roche)",
            "approved_for": "multiple_sclerosis",
            "is_experimental": False
        },
        {
            "name": "Tecfidera (Dimethyl fumarate)",
            "type": TreatmentType.MEDICATION.value,
            "description": "An oral medication that may protect nerve cells and help regulate the immune system.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "240 mg twice daily",
            "administration_route": "Oral",
            "avg_effectiveness": 3.7,
            "side_effects": "Flushing, stomach pain, diarrhea, nausea",
            "contraindications": "Hypersensitivity to dimethyl fumarate",
            "manufacturer": "Biogen",
            "approved_for": "multiple_sclerosis",
            "is_experimental": False
        },
        
        # Crohn's Disease & Ulcerative Colitis Treatments
        {
            "name": "Entyvio (Vedolizumab)",
            "type": TreatmentType.BIOLOGIC.value,
            "description": "A gut-selective biologic that blocks inflammatory cells from entering the GI tract.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "300 mg at weeks 0, 2, and 6, then every 8 weeks",
            "administration_route": "Intravenous infusion",
            "avg_effectiveness": 4.0,
            "side_effects": "Nasopharyngitis, headache, joint pain, nausea",
            "contraindications": "Hypersensitivity to vedolizumab",
            "manufacturer": "Takeda",
            "approved_for": "crohns_disease, ulcerative_colitis",
            "is_experimental": False
        },
        {
            "name": "Mesalamine",
            "type": TreatmentType.MEDICATION.value,
            "description": "An anti-inflammatory drug that acts locally in the gut to reduce inflammation.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "2.4-4.8 g daily in divided doses",
            "administration_route": "Oral, rectal",
            "avg_effectiveness": 3.5,
            "side_effects": "Headache, abdominal pain, nausea, diarrhea",
            "contraindications": "Hypersensitivity to salicylates",
            "manufacturer": "Various",
            "approved_for": "ulcerative_colitis",
            "is_experimental": False
        },
        
        # Psoriasis Treatments
        {
            "name": "Otezla (Apremilast)",
            "type": TreatmentType.MEDICATION.value,
            "description": "A PDE4 inhibitor that reduces inflammation inside cells.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "30 mg twice daily",
            "administration_route": "Oral",
            "avg_effectiveness": 3.6,
            "side_effects": "Diarrhea, nausea, headache, upper respiratory infection",
            "contraindications": "Hypersensitivity to apremilast",
            "manufacturer": "Amgen",
            "approved_for": "psoriasis",
            "is_experimental": False
        },
        {
            "name": "Cosentyx (Secukinumab)",
            "type": TreatmentType.BIOLOGIC.value,
            "description": "A monoclonal antibody that selectively binds to the interleukin-17A cytokine and inhibits its interaction with the IL-17 receptor.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "300 mg at weeks 0, 1, 2, 3, and 4, then every 4 weeks",
            "administration_route": "Subcutaneous injection",
            "avg_effectiveness": 4.4,
            "side_effects": "Nasopharyngitis, diarrhea, upper respiratory infection",
            "contraindications": "Serious hypersensitivity to secukinumab",
            "manufacturer": "Novartis",
            "approved_for": "psoriasis",
            "is_experimental": False
        },
        
        # Type 1 Diabetes Treatments
        {
            "name": "Insulin Therapy",
            "type": TreatmentType.MEDICATION.value,
            "description": "Replacement therapy for the insulin that the pancreas can no longer produce.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "Variable, based on individual needs",
            "administration_route": "Subcutaneous injection or pump",
            "avg_effectiveness": 4.7,
            "side_effects": "Hypoglycemia, weight gain, injection site reactions",
            "contraindications": "None for appropriate doses",
            "manufacturer": "Various",
            "approved_for": "type_1_diabetes",
            "is_experimental": False
        },
        
        # Lupus Treatments
        {
            "name": "Benlysta (Belimumab)",
            "type": TreatmentType.BIOLOGIC.value,
            "description": "A monoclonal antibody that inhibits B-lymphocyte stimulator (BLyS), thereby reducing B cell survival.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "10 mg/kg at 2-week intervals for the first 3 doses, then every 4 weeks",
            "administration_route": "Intravenous infusion or subcutaneous injection",
            "avg_effectiveness": 3.9,
            "side_effects": "Nausea, diarrhea, fever, insomnia, depression",
            "contraindications": "Previous anaphylaxis with belimumab",
            "manufacturer": "GSK",
            "approved_for": "lupus",
            "is_experimental": False
        },
        {
            "name": "Hydroxychloroquine",
            "type": TreatmentType.MEDICATION.value,
            "description": "An antimalarial drug that modulates the immune system and reduces inflammation.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "200-400 mg daily",
            "administration_route": "Oral",
            "avg_effectiveness": 3.7,
            "side_effects": "Retinal toxicity, gastrointestinal upset, skin rash",
            "contraindications": "Retinal or visual field changes, porphyria, G6PD deficiency",
            "manufacturer": "Various",
            "approved_for": "lupus, rheumatoid_arthritis",
            "is_experimental": False
        },
        
        # Celiac Disease Treatments
        {
            "name": "Gluten-Free Diet",
            "type": TreatmentType.DIETARY.value,
            "description": "Strict elimination of gluten-containing foods from the diet.",
            "status": TreatmentStatus.APPROVED.value,
            "dosage_info": "Lifelong adherence required",
            "administration_route": "Dietary",
            "avg_effectiveness": 4.8,
            "side_effects": "Nutritional deficiencies if not properly balanced",
            "contraindications": "None",
            "manufacturer": "N/A",
            "approved_for": "celiac_disease",
            "is_experimental": False
        },
        
        # Experimental Treatments (across conditions)
        {
            "name": "JAK Inhibitor Therapy",
            "type": TreatmentType.EXPERIMENTAL.value,
            "description": "New class of medications that work by inhibiting the activity of janus kinase enzymes, which are involved in immune cell signaling.",
            "status": TreatmentStatus.EXPERIMENTAL.value,
            "dosage_info": "Varies by specific medication",
            "administration_route": "Oral",
            "avg_effectiveness": 4.1,
            "side_effects": "Increased risk of infections, blood clots, elevated cholesterol",
            "contraindications": "Active infections, pregnancy",
            "manufacturer": "Various",
            "approved_for": "rheumatoid_arthritis, psoriasis",
            "is_experimental": True
        },
        {
            "name": "Mesenchymal Stem Cell Therapy",
            "type": TreatmentType.EXPERIMENTAL.value,
            "description": "Stem cell therapy aimed at modulating the immune system and promoting tissue repair.",
            "status": TreatmentStatus.EXPERIMENTAL.value,
            "dosage_info": "Protocol dependent",
            "administration_route": "Intravenous infusion or local injection",
            "avg_effectiveness": 3.5,
            "side_effects": "Infection, immune reactions, potential tumorigenicity",
            "contraindications": "Active malignancy, pregnancy",
            "manufacturer": "Various research institutions",
            "approved_for": "multiple_sclerosis, crohns_disease, lupus",
            "is_experimental": True
        },
        {
            "name": "Fecal Microbiota Transplantation",
            "type": TreatmentType.EXPERIMENTAL.value,
            "description": "Transfer of fecal matter from a healthy donor to restore the gut microbiome balance.",
            "status": TreatmentStatus.EXPERIMENTAL.value,
            "dosage_info": "Single or multiple administrations",
            "administration_route": "Colonoscopy, enema, or oral capsules",
            "avg_effectiveness": 3.8,
            "side_effects": "Gastrointestinal discomfort, potential infection transmission",
            "contraindications": "Compromised immune system, certain infections",
            "manufacturer": "N/A",
            "approved_for": "ulcerative_colitis, crohns_disease",
            "is_experimental": True
        }
    ]
    
    # Create and add treatments to database
    created_treatments = []
    for treatment_data in treatments_data:
        # Check if treatment already exists
        existing_treatment = db.query(Treatment).filter(Treatment.name == treatment_data["name"]).first()
        
        if existing_treatment:
            logger.info(f"Treatment '{treatment_data['name']}' already exists")
            created_treatments.append(existing_treatment)
        else:
            # Create new treatment
            new_treatment = Treatment(**treatment_data)
            db.add(new_treatment)
            created_treatments.append(new_treatment)
            logger.info(f"Added treatment: {treatment_data['name']}")
    
    # Commit changes to database
    db.commit()
    
    logger.info(f"Loaded {len(created_treatments)} treatments")
    return created_treatments

def load_clinical_trials(db: Session) -> List[ClinicalTrial]:
    """
    Load sample clinical trials into the database.
    
    Args:
        db: Database session
        
    Returns:
        List of created clinical trial objects
    """
    logger.info("Loading sample clinical trials...")
    
    # Get treatments to reference in trials
    treatments = {t.name: t for t in db.query(Treatment).all()}
    
    # Define sample clinical trials
    now = datetime.now()
    trials_data = [
        {
            "nct_id": "NCT01234567",
            "title": "Efficacy and Safety of JAK Inhibitors in Rheumatoid Arthritis",
            "description": "A Phase 3 trial investigating the efficacy and safety of JAK inhibitors in patients with moderate to severe rheumatoid arthritis who have had an inadequate response to conventional DMARDs.",
            "phase": TrialPhase.PHASE_3.value,
            "status": TrialStatus.RECRUITING.value,
            "conditions": "rheumatoid_arthritis",
            "inclusion_criteria": "Adults 18-75 with moderate to severe rheumatoid arthritis; Inadequate response to at least one conventional DMARD",
            "exclusion_criteria": "Pregnancy or breastfeeding; Active infection; History of thromboembolic events",
            "start_date": now - timedelta(days=30),
            "completion_date": now + timedelta(days=365),
            "locations": json.dumps([
                {"name": "University Medical Center", "city": "New York", "state": "NY", "country": "USA"},
                {"name": "Research Hospital", "city": "Chicago", "state": "IL", "country": "USA"}
            ]),
            "contact_name": "John Smith, MD",
            "contact_email": "jsmith@researchcenter.org",
            "contact_phone": "555-123-4567",
            "treatment_id": treatments.get("JAK Inhibitor Therapy").id if "JAK Inhibitor Therapy" in treatments else None
        },
        {
            "nct_id": "NCT02345678",
            "title": "Mesenchymal Stem Cell Therapy for Multiple Sclerosis",
            "description": "A Phase 2 trial evaluating the safety and efficacy of autologous mesenchymal stem cell transplantation in patients with relapsing-remitting multiple sclerosis.",
            "phase": TrialPhase.PHASE_2.value,
            "status": TrialStatus.RECRUITING.value,
            "conditions": "multiple_sclerosis",
            "inclusion_criteria": "Adults 18-60 with relapsing-remitting multiple sclerosis; EDSS score between 2.0 and 6.5",
            "exclusion_criteria": "Pregnancy or breastfeeding; Active infection; Primary progressive MS",
            "start_date": now - timedelta(days=60),
            "completion_date": now + timedelta(days=730),
            "locations": json.dumps([
                {"name": "Stem Cell Research Center", "city": "Boston", "state": "MA", "country": "USA"},
                {"name": "University Hospital", "city": "San Francisco", "state": "CA", "country": "USA"}
            ]),
            "contact_name": "Sarah Johnson, PhD",
            "contact_email": "sjohnson@stemcellcenter.org",
            "contact_phone": "555-987-6543",
            "treatment_id": treatments.get("Mesenchymal Stem Cell Therapy").id if "Mesenchymal Stem Cell Therapy" in treatments else None
        },
        {
            "nct_id": "NCT03456789",
            "title": "Fecal Microbiota Transplantation for Ulcerative Colitis",
            "description": "A Phase 2 trial investigating the efficacy of fecal microbiota transplantation in inducing clinical remission in patients with moderate to severe ulcerative colitis.",
            "phase": TrialPhase.PHASE_2.value,
            "status": TrialStatus.RECRUITING.value,
            "conditions": "ulcerative_colitis",
            "inclusion_criteria": "Adults 18-70 with moderate to severe ulcerative colitis; Mayo score ≥ 6",
            "exclusion_criteria": "Pregnancy or breastfeeding; Severe comorbidities; Use of antibiotics within 3 months",
            "start_date": now - timedelta(days=90),
            "completion_date": now + timedelta(days=365),
            "locations": json.dumps([
                {"name": "Gastroenterology Research Institute", "city": "Seattle", "state": "WA", "country": "USA"},
                {"name": "Digestive Health Center", "city": "Miami", "state": "FL", "country": "USA"}
            ]),
            "contact_name": "Robert Brown, MD",
            "contact_email": "rbrown@digestivehealth.org",
            "contact_phone": "555-222-3333",
            "treatment_id": treatments.get("Fecal Microbiota Transplantation").id if "Fecal Microbiota Transplantation" in treatments else None
        },
        {
            "nct_id": "NCT04567890",
            "title": "Novel Biologic Therapy for Lupus",
            "description": "A Phase 2 trial evaluating the safety and efficacy of a novel biologic targeting type I interferons in patients with systemic lupus erythematosus.",
            "phase": TrialPhase.PHASE_2.value,
            "status": TrialStatus.RECRUITING.value,
            "conditions": "lupus",
            "inclusion_criteria": "Adults 18-65 with SLE; SLEDAI score ≥ 6; Positive ANA or anti-dsDNA antibodies",
            "exclusion_criteria": "Pregnancy or breastfeeding; Severe active lupus nephritis; Active CNS lupus",
            "start_date": now - timedelta(days=45),
            "completion_date": now + timedelta(days=548),
            "locations": json.dumps([
                {"name": "Autoimmune Research Center", "city": "Philadelphia", "state": "PA", "country": "USA"},
                {"name": "Rheumatology Institute", "city": "Houston", "state": "TX", "country": "USA"}
            ]),
            "contact_name": "Lisa Chen, MD",
            "contact_email": "lchen@autoimmune.org",
            "contact_phone": "555-444-5555",
            "treatment_id": None  # No specific treatment reference for this trial
        },
        {
            "nct_id": "NCT05678901",
            "title": "Combination Therapy for Psoriasis",
            "description": "A Phase 3 trial evaluating the efficacy and safety of combining biologic therapy with phototherapy for moderate to severe plaque psoriasis.",
            "phase": TrialPhase.PHASE_3.value,
            "status": TrialStatus.RECRUITING.value,
            "conditions": "psoriasis",
            "inclusion_criteria": "Adults 18-75 with moderate to severe plaque psoriasis; PASI score ≥ 12; BSA involvement ≥ 10%",
            "exclusion_criteria": "Pregnancy or breastfeeding; History of skin cancer; Photosensitivity disorders",
            "start_date": now - timedelta(days=15),
            "completion_date": now + timedelta(days=456),
            "locations": json.dumps([
                {"name": "Dermatology Research Center", "city": "Los Angeles", "state": "CA", "country": "USA"},
                {"name": "Skin Health Institute", "city": "Denver", "state": "CO", "country": "USA"}
            ]),
            "contact_name": "Michael Wong, MD",
            "contact_email": "mwong@dermatology.org",
            "contact_phone": "555-666-7777",
            "treatment_id": treatments.get("Cosentyx (Secukinumab)").id if "Cosentyx (Secukinumab)" in treatments else None
        }
    ]
    
    # Create and add clinical trials to database
    created_trials = []
    for trial_data in trials_data:
        # Check if trial already exists
        existing_trial = db.query(ClinicalTrial).filter(ClinicalTrial.nct_id == trial_data["nct_id"]).first()
        
        if existing_trial:
            logger.info(f"Clinical trial '{trial_data['nct_id']}' already exists")
            created_trials.append(existing_trial)
        else:
            # Create new clinical trial
            new_trial = ClinicalTrial(**trial_data)
            db.add(new_trial)
            created_trials.append(new_trial)
            logger.info(f"Added clinical trial: {trial_data['nct_id']} - {trial_data['title']}")
    
    # Commit changes to database
    db.commit()
    
    logger.info(f"Loaded {len(created_trials)} clinical trials")
    return created_trials

def create_sample_patients(db: Session) -> List[Patient]:
    """
    Create sample patients in the database.
    
    Args:
        db: Database session
        
    Returns:
        List of created patient objects
    """
    logger.info("Creating sample patients...")
    
    # Define sample patients
    patients_data = [
        {
            "name": "John Doe",
            "age": 45,
            "gender": "Male",
            "weight": 82.5,
            "height": 178.0,
            "disease_type": AutoimmuneDisease.RHEUMATOID_ARTHRITIS.value,
            "diagnosis_date": datetime.now() - timedelta(days=365*3),  # 3 years ago
            "symptom_severity": SymptomSeverity.MODERATE.value,
            "medical_history": "Hypertension, controlled with medication. No history of other autoimmune conditions.",
            "allergies": "Penicillin"
        },
        {
            "name": "Jane Smith",
            "age": 32,
            "gender": "Female",
            "weight": 65.0,
            "height": 165.0,
            "disease_type": AutoimmuneDisease.LUPUS.value,
            "diagnosis_date": datetime.now() - timedelta(days=365*2),  # 2 years ago
            "symptom_severity": SymptomSeverity.SEVERE.value,
            "medical_history": "Raynaud's phenomenon, photosensitivity. Family history of autoimmune disease.",
            "allergies": "None known"
        },
        {
            "name": "Robert Johnson",
            "age": 56,
            "gender": "Male",
            "weight": 90.0,
            "height": 182.0,
            "disease_type": AutoimmuneDisease.CROHNS_DISEASE.value,
            "diagnosis_date": datetime.now() - timedelta(days=365*5),  # 5 years ago
            "symptom_severity": SymptomSeverity.VERY_SEVERE.value,
            "medical_history": "Prior bowel resection surgery. Anemia due to chronic blood loss.",
            "allergies": "Sulfa drugs"
        },
        {
            "name": "Maria Garcia",
            "age": 28,
            "gender": "Female",
            "weight": 58.0,
            "height": 160.0,
            "disease_type": AutoimmuneDisease.MULTIPLE_SCLEROSIS.value,
            "diagnosis_date": datetime.now() - timedelta(days=365*1.5),  # 1.5 years ago
            "symptom_severity": SymptomSeverity.MODERATE.value,
            "medical_history": "Optic neuritis at initial presentation. No other significant medical history.",
            "allergies": "Latex"
        },
        {
            "name": "David Chen",
            "age": 41,
            "gender": "Male",
            "weight": 75.0,
            "height": 175.0,
            "disease_type": AutoimmuneDisease.PSORIASIS.value,
            "diagnosis_date": datetime.now() - timedelta(days=365*10),  # 10 years ago
            "symptom_severity": SymptomSeverity.MILD.value,
            "medical_history": "Psoriatic nail involvement. Otherwise healthy.",
            "allergies": "None known"
        }
    ]
    
    # Create and add patients to database
    created_patients = []
    for patient_data in patients_data:
        # Check if patient already exists
        existing_patient = db.query(Patient).filter(Patient.name == patient_data["name"]).first()
        
        if existing_patient:
            logger.info(f"Patient '{patient_data['name']}' already exists")
            created_patients.append(existing_patient)
        else:
            # Create new patient
            new_patient = Patient(**patient_data)
            db.add(new_patient)
            created_patients.append(new_patient)
            logger.info(f"Added patient: {patient_data['name']}")
    
    # Commit changes to database
    db.commit()
    
    logger.info(f"Created {len(created_patients)} sample patients")
    return created_patients

def create_sample_treatment_logs(db: Session) -> List[TreatmentLog]:
    """
    Create sample treatment logs for patients.
    
    Args:
        db: Database session
        
    Returns:
        List of created treatment log objects
    """
    logger.info("Creating sample treatment logs...")
    
    # Get patients and treatments
    patients = db.query(Patient).all()
    treatments = db.query(Treatment).all()
    
    if not patients or not treatments:
        logger.warning("No patients or treatments found. Cannot create treatment logs.")
        return []
    
    # Map treatments by disease type for easier access
    treatment_map = {}
    for treatment in treatments:
        for disease in treatment.approved_for.split(','):
            disease = disease.strip()
            if disease not in treatment_map:
                treatment_map[disease] = []
            treatment_map[disease].append(treatment)
    
    # Create treatment logs
    created_logs = []
    for patient in patients:
        # Get relevant treatments for this patient's condition
        relevant_treatments = treatment_map.get(patient.disease_type, [])
        
        if not relevant_treatments:
            continue
            
        # Create 1-3 treatment logs per patient
        for i in range(min(len(relevant_treatments), 3)):
            treatment = relevant_treatments[i]
            
            # Define start date (between diagnosis date and now)
            days_since_diagnosis = (datetime.now() - patient.diagnosis_date).days
            start_date = patient.diagnosis_date + timedelta(days=max(30, days_since_diagnosis // (i+2)))
            
            # Define end date (some treatments ongoing, some ended)
            end_date = None
            if i == 0:  # First treatment (ended)
                end_date = start_date + timedelta(days=90)
                is_active = False
            else:
                is_active = True
            
            # Create log
            log_data = {
                "patient_id": patient.id,
                "treatment_id": treatment.id,
                "start_date": start_date,
                "end_date": end_date,
                "dosage": f"Standard dosage per {treatment.dosage_info}",
                "frequency": "As prescribed",
                "effectiveness_rating": min(5, max(1, int(treatment.avg_effectiveness) + (i % 2 - 1))),  # Vary slightly from treatment average
                "side_effects_experienced": "Mild " + treatment.side_effects.split(',')[0] if treatment.side_effects else "None",
                "notes": f"Patient {['tolerated well', 'had some initial difficulties but adjusted', 'reported improvement after 2 weeks'][i % 3]}",
                "is_active": is_active
            }
            
            new_log = TreatmentLog(**log_data)
            db.add(new_log)
            created_logs.append(new_log)
            logger.info(f"Added treatment log: {patient.name} - {treatment.name}")
    
    # Commit changes to database
    db.commit()
    
    logger.info(f"Created {len(created_logs)} sample treatment logs")
    return created_logs

def initialize_sample_data(db: Session) -> None:
    """
    Initialize the database with a complete set of sample data.
    
    This function creates treatments, clinical trials, patients, and treatment logs.
    It's a complete initialization for a demonstration system.
    
    Args:
        db: Database session
    """
    try:
        logger.info("Initializing sample data...")
        
        # Load treatments and clinical trials
        load_treatments(db)
        load_clinical_trials(db)
        
        # Create patients and treatment logs
        create_sample_patients(db)
        create_sample_treatment_logs(db)
        
        logger.info("Sample data initialization complete.")
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")
        db.rollback()
        raise 