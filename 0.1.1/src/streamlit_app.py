"""
Streamlit application for FixImmune - Autoimmune Treatment Recommendation System
"""

import os
import sys
import json
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import requests
from dotenv import load_dotenv

# Setup path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "http://localhost:8000/api/v1"

# Set page configuration
st.set_page_config(
    page_title="FixImmune - Autoimmune Treatment Recommendations",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define helper functions
def fetch_patients():
    """Fetch patients from the API."""
    try:
        response = requests.get(f"{API_URL}/patients/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        return []

def fetch_treatments():
    """Fetch treatments from the API."""
    try:
        response = requests.get(f"{API_URL}/treatments/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching treatments: {e}")
        return []

def fetch_treatment_logs(patient_id=None):
    """Fetch treatment logs from the API."""
    try:
        url = f"{API_URL}/treatment-logs/"
        if patient_id:
            url += f"?patient_id={patient_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching treatment logs: {e}")
        return []

def fetch_clinical_trials(condition=None):
    """Fetch clinical trials from the API."""
    try:
        url = f"{API_URL}/clinical-trials/"
        if condition:
            url += f"?condition={condition}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching clinical trials: {e}")
        return []

def fetch_recommendations(patient_id, include_experimental=False, include_clinical_trials=False):
    """Fetch treatment recommendations for a patient."""
    try:
        url = f"{API_URL}/recommendations/{patient_id}?include_experimental={str(include_experimental).lower()}&include_clinical_trials={str(include_clinical_trials).lower()}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        return {"recommendations": []}

def fetch_patient(patient_id):
    """Fetch a specific patient by ID."""
    try:
        response = requests.get(f"{API_URL}/patients/{patient_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {e}")
        return None

def create_patient(patient_data):
    """Create a new patient."""
    try:
        response = requests.post(
            f"{API_URL}/patients/",
            json=patient_data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error creating patient: {e}")
        return None

def create_treatment_log(log_data):
    """Create a new treatment log entry."""
    try:
        response = requests.post(
            f"{API_URL}/treatment-logs/",
            json=log_data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error creating treatment log: {e}")
        return None

# Sidebar for navigation
st.sidebar.title("FixImmune")
st.sidebar.image("https://img.icons8.com/color/96/000000/healthcare-and-medical.png")
st.sidebar.caption("Autoimmune Treatment Recommendation System")

# Navigation menu
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Patient Dashboard", "Treatment Recommendations", "Clinical Trials", "Treatment Tracking"]
)

# Display appropriate page based on selection
if page == "Home":
    st.title("Welcome to FixImmune")
    st.subheader("Personalized Autoimmune Treatment Recommendation System")
    
    st.markdown("""
    FixImmune is an intelligent tool designed to help autoimmune patients and their healthcare providers 
    identify the most effective treatments based on their specific disease characteristics.
    
    ### Key Features:
    
    - **Personalized Treatment Recommendations** - Receive treatment suggestions tailored to your specific condition and symptoms
    - **Treatment Effectiveness Tracking** - Log and monitor your treatment progress over time
    - **Clinical Trial Access** - Discover relevant clinical trials that might be suitable for your condition
    - **Data-Driven Insights** - Leverage real-world effectiveness data to inform your treatment decisions
    
    ### Getting Started:
    
    1. Create your patient profile
    2. Explore personalized treatment recommendations
    3. Track your treatment progress
    4. Discover relevant clinical trials
    
    *Note: This is an MVP (Minimum Viable Product) version. Always consult with your healthcare provider before making any treatment decisions.*
    """)
    
    st.info("FixImmune is a prototype application and not a substitute for professional medical advice.")
    
elif page == "Patient Dashboard":
    st.title("Patient Dashboard")
    
    tab1, tab2 = st.tabs(["Patient List", "Create New Patient"])
    
    with tab1:
        st.subheader("Registered Patients")
        
        patients = fetch_patients()
        
        if patients:
            # Create a DataFrame for display
            patients_df = pd.DataFrame(patients)
            patients_df['diagnosis_date'] = pd.to_datetime(patients_df['diagnosis_date']).dt.date
            patients_df['created_at'] = pd.to_datetime(patients_df['created_at']).dt.date
            
            # Display the patient list
            st.dataframe(
                patients_df[['name', 'age', 'gender', 'disease_type', 'symptom_severity', 'diagnosis_date']],
                column_config={
                    "name": "Patient Name",
                    "age": "Age",
                    "gender": "Gender",
                    "disease_type": "Condition",
                    "symptom_severity": "Symptom Severity (1-5)",
                    "diagnosis_date": "Diagnosis Date"
                },
                use_container_width=True
            )
            
            # Select a patient to view details
            selected_patient_id = st.selectbox(
                "Select a patient to view details:",
                options=[p["id"] for p in patients],
                format_func=lambda x: next((p["name"] for p in patients if p["id"] == x), x)
            )
            
            if selected_patient_id:
                patient = next((p for p in patients if p["id"] == selected_patient_id), None)
                
                if patient:
                    st.subheader(f"Patient Details: {patient['name']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Information**")
                        st.write(f"Age: {patient['age']}")
                        st.write(f"Gender: {patient['gender']}")
                        st.write(f"Weight: {patient['weight']} kg" if patient['weight'] else "Weight: Not recorded")
                        st.write(f"Height: {patient['height']} cm" if patient['height'] else "Height: Not recorded")
                    
                    with col2:
                        st.write("**Medical Information**")
                        st.write(f"Condition: {patient['disease_type'].replace('_', ' ').title()}")
                        st.write(f"Symptom Severity: {patient['symptom_severity']}/5")
                        st.write(f"Diagnosis Date: {patient['diagnosis_date']}")
                    
                    st.write("**Medical History**")
                    st.write(patient['medical_history'] if patient['medical_history'] else "No medical history recorded")
                    
                    st.write("**Allergies**")
                    st.write(patient['allergies'] if patient['allergies'] else "No allergies recorded")
                    
                    # Treatment History
                    st.subheader("Treatment History")
                    treatment_logs = fetch_treatment_logs(patient_id=patient['id'])
                    
                    if treatment_logs:
                        logs_df = pd.DataFrame(treatment_logs)
                        logs_df['start_date'] = pd.to_datetime(logs_df['start_date']).dt.date
                        logs_df['end_date'] = pd.to_datetime(logs_df['end_date']).dt.date
                        
                        # Fetch treatment names
                        treatments = fetch_treatments()
                        treatment_map = {t['id']: t['name'] for t in treatments}
                        
                        logs_df['treatment_name'] = logs_df['treatment_id'].map(treatment_map)
                        
                        st.dataframe(
                            logs_df[['treatment_name', 'start_date', 'end_date', 'effectiveness_rating', 'is_active']],
                            column_config={
                                "treatment_name": "Treatment",
                                "start_date": "Started",
                                "end_date": "Ended",
                                "effectiveness_rating": "Effectiveness (1-5)",
                                "is_active": "Active"
                            },
                            use_container_width=True
                        )
                        
                        # Show effectiveness chart
                        if len(logs_df) > 0:
                            st.subheader("Treatment Effectiveness")
                            
                            # Prepare data for chart
                            chart_data = logs_df[['treatment_name', 'effectiveness_rating']].sort_values('effectiveness_rating', ascending=False)
                            
                            # Create bar chart
                            fig = px.bar(
                                chart_data,
                                x='treatment_name',
                                y='effectiveness_rating',
                                labels={'treatment_name': 'Treatment', 'effectiveness_rating': 'Effectiveness Rating (1-5)'},
                                title='Treatment Effectiveness Comparison',
                                color='effectiveness_rating',
                                color_continuous_scale=px.colors.sequential.Viridis
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No treatment history recorded for this patient.")
        else:
            st.info("No patients registered yet. Create a new patient to get started.")
    
    with tab2:
        st.subheader("Create New Patient")
        
        with st.form("new_patient_form"):
            # Basic information
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Patient Name")
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            
            with col2:
                weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
                height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
            
            # Disease information
            st.subheader("Medical Information")
            
            disease_options = [
                "rheumatoid_arthritis",
                "lupus",
                "multiple_sclerosis",
                "type_1_diabetes",
                "psoriasis",
                "crohns_disease",
                "ulcerative_colitis",
                "celiac_disease",
                "graves_disease",
                "hashimotos_thyroiditis",
                "other"
            ]
            
            disease_type = st.selectbox(
                "Autoimmune Condition",
                options=disease_options,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            diagnosis_date = st.date_input("Diagnosis Date")
            symptom_severity = st.slider("Symptom Severity", min_value=1, max_value=5, value=3, 
                                        help="1 = Mild, 5 = Debilitating")
            
            medical_history = st.text_area("Medical History")
            allergies = st.text_area("Allergies")
            
            submit_button = st.form_submit_button("Create Patient")
            
            if submit_button:
                if not name:
                    st.error("Patient name is required.")
                else:
                    # Format data for API
                    patient_data = {
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "weight": weight,
                        "height": height,
                        "disease_type": disease_type,
                        "diagnosis_date": diagnosis_date.isoformat() if diagnosis_date else None,
                        "symptom_severity": symptom_severity,
                        "medical_history": medical_history,
                        "allergies": allergies
                    }
                    
                    # Create patient
                    new_patient = create_patient(patient_data)
                    
                    if new_patient:
                        st.success(f"Patient {name} created successfully!")
                        st.balloons()
                    else:
                        st.error("Failed to create patient. Please try again.")

elif page == "Treatment Recommendations":
    st.title("Treatment Recommendations")
    
    # Get list of patients
    patients = fetch_patients()
    
    if not patients:
        st.info("No patients registered yet. Create a patient first to get recommendations.")
    else:
        # Select a patient
        selected_patient_id = st.selectbox(
            "Select a patient:",
            options=[p["id"] for p in patients],
            format_func=lambda x: next((p["name"] for p in patients if p["id"] == x), x)
        )
        
        patient = next((p for p in patients if p["id"] == selected_patient_id), None)
        
        if patient:
            st.write(f"Generating recommendations for **{patient['name']}** with **{patient['disease_type'].replace('_', ' ').title()}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_experimental = st.checkbox("Include Experimental Treatments", value=False)
            
            with col2:
                include_clinical_trials = st.checkbox("Include Clinical Trials", value=True)
            
            # Fetch recommendations
            recommendations = fetch_recommendations(
                patient_id=patient['id'],
                include_experimental=include_experimental,
                include_clinical_trials=include_clinical_trials
            )
            
            # Display recommendations
            st.subheader("Recommended Treatments")
            
            if "recommendations" in recommendations and recommendations["recommendations"]:
                # Create a DataFrame for treatments
                treatments_df = pd.DataFrame(recommendations["recommendations"])
                
                # Display as a table
                st.dataframe(
                    treatments_df[['name', 'type', 'avg_effectiveness', 'is_experimental', 'source']],
                    column_config={
                        "name": "Treatment Name",
                        "type": "Type",
                        "avg_effectiveness": st.column_config.ProgressColumn(
                            "Effectiveness",
                            min_value=0,
                            max_value=5,
                            format="%f/5"
                        ),
                        "is_experimental": "Experimental",
                        "source": "Recommendation Source"
                    },
                    use_container_width=True
                )
                
                # Create a visualization
                st.subheader("Effectiveness Comparison")
                
                fig = px.bar(
                    treatments_df,
                    x='name',
                    y='avg_effectiveness',
                    color='is_experimental',
                    labels={'name': 'Treatment', 'avg_effectiveness': 'Avg. Effectiveness (1-5)', 'is_experimental': 'Experimental'},
                    title='Recommended Treatments by Effectiveness',
                    barmode='group',
                    color_discrete_map={True: 'orange', False: 'blue'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # For each treatment, provide more details
                st.subheader("Treatment Details")
                
                for treatment_idx, treatment in enumerate(recommendations["recommendations"]):
                    with st.expander(f"{treatment['name']} - {treatment['type'].replace('_', ' ').title()}"):
                        st.write(f"**Effectiveness Rating:** {treatment['avg_effectiveness']}/5")
                        st.write(f"**Experimental:** {'Yes' if treatment['is_experimental'] else 'No'}")
                        st.write(f"**Recommendation Source:** {treatment['source'].replace('_', ' ').title()}")
                        st.write(f"**Confidence Score:** {treatment['confidence_score'] * 100:.1f}%")
                        
                        # Add a button to log this treatment
                        if st.button(f"Log This Treatment", key=f"log_{treatment_idx}"):
                            st.session_state['selected_treatment'] = treatment
                            st.session_state['selected_patient'] = patient
                            st.experimental_rerun()
                
                # If a treatment is selected to log
                if 'selected_treatment' in st.session_state and 'selected_patient' in st.session_state:
                    st.subheader(f"Log Treatment: {st.session_state['selected_treatment']['name']}")
                    
                    with st.form("log_treatment_form"):
                        start_date = st.date_input("Start Date", value=datetime.now().date())
                        dosage = st.text_input("Dosage (e.g., 10mg twice daily)")
                        frequency = st.text_input("Frequency (e.g., daily, weekly)")
                        notes = st.text_area("Notes")
                        
                        submit_button = st.form_submit_button("Log Treatment")
                        
                        if submit_button:
                            # Create treatment log entry
                            log_data = {
                                "patient_id": st.session_state['selected_patient']['id'],
                                "treatment_id": st.session_state['selected_treatment']['id'],
                                "start_date": start_date.isoformat(),
                                "dosage": dosage,
                                "frequency": frequency,
                                "notes": notes,
                                "is_active": True,
                                "effectiveness_rating": 3  # Default to moderate improvement
                            }
                            
                            # Create log
                            new_log = create_treatment_log(log_data)
                            
                            if new_log:
                                st.success(f"Treatment logged successfully!")
                                # Clear session state
                                del st.session_state['selected_treatment']
                                del st.session_state['selected_patient']
                            else:
                                st.error("Failed to log treatment. Please try again.")
            else:
                st.info("No treatment recommendations found for this patient.")
            
            # Display clinical trials if requested
            if include_clinical_trials and "clinical_trials" in recommendations:
                st.subheader("Relevant Clinical Trials")
                
                if recommendations["clinical_trials"]:
                    # Create a DataFrame for trials
                    trials_df = pd.DataFrame(recommendations["clinical_trials"])
                    
                    # Display as a table
                    st.dataframe(
                        trials_df[['title', 'phase', 'status']],
                        column_config={
                            "title": "Trial Name",
                            "phase": "Phase",
                            "status": "Status"
                        },
                        use_container_width=True
                    )
                    
                    # For each trial, provide more details
                    for trial_idx, trial in enumerate(recommendations["clinical_trials"]):
                        with st.expander(f"{trial['title']} - Phase {trial['phase']}"):
                            st.write(f"**NCT ID:** {trial['nct_id']}")
                            st.write(f"**Status:** {trial['status'].replace('_', ' ').title()}")
                            st.write(f"**Description:** {trial['description']}")
                            
                            # Display locations if available
                            if 'locations' in trial and trial['locations']:
                                try:
                                    locations = json.loads(trial['locations'])
                                    st.write("**Locations:**")
                                    for loc in locations:
                                        st.write(f"- {loc['name']}, {loc['city']}, {loc['state']}, {loc['country']}")
                                except:
                                    st.write(f"**Locations:** {trial['locations']}")
                            
                            # Add dates
                            st.write(f"**Start Date:** {trial['start_date']}")
                            if 'completion_date' in trial and trial['completion_date']:
                                st.write(f"**Expected Completion:** {trial['completion_date']}")
                else:
                    st.info("No relevant clinical trials found for this patient.")

elif page == "Clinical Trials":
    st.title("Clinical Trials")
    
    # Search options
    st.subheader("Search Clinical Trials")
    
    # Filter options
    condition = st.text_input("Condition", placeholder="e.g., rheumatoid_arthritis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        status = st.selectbox(
            "Status",
            options=["", "recruiting", "active_not_recruiting", "completed", "withdrawn", "suspended", "terminated", "unknown"],
            format_func=lambda x: x.replace('_', ' ').title() if x else "All"
        )
    
    with col2:
        phase = st.selectbox(
            "Phase",
            options=["", "phase_1", "phase_2", "phase_3", "phase_4", "not_applicable"],
            format_func=lambda x: x.replace('_', ' ').title() if x else "All"
        )
    
    # Fetch trials
    if st.button("Search"):
        # Build query parameters
        params = {}
        if condition:
            params["condition"] = condition
        if status:
            params["status"] = status
        if phase:
            params["phase"] = phase
        
        # Fetch trials
        trials = fetch_clinical_trials(**params)
        
        if trials:
            st.success(f"Found {len(trials)} clinical trials")
            
            # Display trials
            trials_df = pd.DataFrame(trials)
            
            # Format dates
            if 'start_date' in trials_df.columns:
                trials_df['start_date'] = pd.to_datetime(trials_df['start_date']).dt.date
            if 'completion_date' in trials_df.columns:
                trials_df['completion_date'] = pd.to_datetime(trials_df['completion_date']).dt.date
            
            # Display as a table
            st.dataframe(
                trials_df[['title', 'phase', 'status', 'start_date']],
                column_config={
                    "title": "Trial Name",
                    "phase": "Phase",
                    "status": "Status",
                    "start_date": "Start Date"
                },
                use_container_width=True
            )
            
            # For each trial, provide more details
            for trial_idx, trial in enumerate(trials):
                with st.expander(f"{trial['title']} - Phase {trial['phase']}"):
                    st.write(f"**NCT ID:** {trial['nct_id']}")
                    st.write(f"**Status:** {trial['status'].replace('_', ' ').title()}")
                    st.write(f"**Description:** {trial['description']}")
                    
                    # Display conditions
                    st.write(f"**Conditions:** {trial['conditions']}")
                    
                    # Display locations if available
                    if 'locations' in trial and trial['locations']:
                        try:
                            locations = json.loads(trial['locations'])
                            st.write("**Locations:**")
                            for loc in locations:
                                st.write(f"- {loc['name']}, {loc['city']}, {loc['state']}, {loc['country']}")
                        except:
                            st.write(f"**Locations:** {trial['locations']}")
                    
                    # Add dates
                    st.write(f"**Start Date:** {trial['start_date']}")
                    if 'completion_date' in trial and trial['completion_date']:
                        st.write(f"**Expected Completion:** {trial['completion_date']}")
                    
                    # Add inclusion/exclusion criteria if available
                    if 'inclusion_criteria' in trial and trial['inclusion_criteria']:
                        st.write("**Inclusion Criteria:**")
                        st.write(trial['inclusion_criteria'])
                    
                    if 'exclusion_criteria' in trial and trial['exclusion_criteria']:
                        st.write("**Exclusion Criteria:**")
                        st.write(trial['exclusion_criteria'])
                    
                    # Add contact information if available
                    if 'contact_name' in trial and trial['contact_name']:
                        st.write("**Contact Information:**")
                        st.write(f"Name: {trial['contact_name']}")
                        if 'contact_email' in trial and trial['contact_email']:
                            st.write(f"Email: {trial['contact_email']}")
                        if 'contact_phone' in trial and trial['contact_phone']:
                            st.write(f"Phone: {trial['contact_phone']}")
        else:
            st.info("No clinical trials found matching your criteria.")

elif page == "Treatment Tracking":
    st.title("Treatment Tracking")
    
    # Get list of patients
    patients = fetch_patients()
    
    if not patients:
        st.info("No patients registered yet. Create a patient first to track treatments.")
    else:
        # Select a patient
        selected_patient_id = st.selectbox(
            "Select a patient:",
            options=[p["id"] for p in patients],
            format_func=lambda x: next((p["name"] for p in patients if p["id"] == x), x)
        )
        
        patient = next((p for p in patients if p["id"] == selected_patient_id), None)
        
        if patient:
            st.write(f"Treatment history for **{patient['name']}**")
            
            # Fetch treatment logs
            treatment_logs = fetch_treatment_logs(patient_id=patient['id'])
            
            if treatment_logs:
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Treatment History", "Treatment Effectiveness", "Log New Treatment"])
                
                with tab1:
                    st.subheader("Treatment History")
                    
                    # Convert to DataFrame
                    logs_df = pd.DataFrame(treatment_logs)
                    logs_df['start_date'] = pd.to_datetime(logs_df['start_date']).dt.date
                    logs_df['end_date'] = pd.to_datetime(logs_df['end_date']).dt.date
                    
                    # Fetch treatment names
                    treatments = fetch_treatments()
                    treatment_map = {t['id']: t['name'] for t in treatments}
                    
                    logs_df['treatment_name'] = logs_df['treatment_id'].map(treatment_map)
                    
                    # Display as a table
                    st.dataframe(
                        logs_df[['treatment_name', 'start_date', 'end_date', 'effectiveness_rating', 'is_active']],
                        column_config={
                            "treatment_name": "Treatment",
                            "start_date": "Started",
                            "end_date": "Ended",
                            "effectiveness_rating": "Effectiveness (1-5)",
                            "is_active": "Active"
                        },
                        use_container_width=True
                    )
                    
                    # For each log, provide more details
                    for log_idx, log in enumerate(treatment_logs):
                        treatment_name = treatment_map.get(log['treatment_id'], log['treatment_id'])
                        
                        with st.expander(f"{treatment_name} - Started {log['start_date']}"):
                            st.write(f"**Treatment:** {treatment_name}")
                            st.write(f"**Started:** {log['start_date']}")
                            if log['end_date']:
                                st.write(f"**Ended:** {log['end_date']}")
                            else:
                                st.write("**Status:** Ongoing")
                            
                            st.write(f"**Effectiveness Rating:** {log['effectiveness_rating']}/5")
                            st.write(f"**Dosage:** {log['dosage'] if log['dosage'] else 'Not specified'}")
                            st.write(f"**Frequency:** {log['frequency'] if log['frequency'] else 'Not specified'}")
                            
                            if log['side_effects_experienced']:
                                st.write(f"**Side Effects:** {log['side_effects_experienced']}")
                            
                            if log['notes']:
                                st.write(f"**Notes:** {log['notes']}")
                
                with tab2:
                    st.subheader("Treatment Effectiveness")
                    
                    # Convert to DataFrame
                    logs_df = pd.DataFrame(treatment_logs)
                    
                    # Fetch treatment names
                    treatments = fetch_treatments()
                    treatment_map = {t['id']: t['name'] for t in treatments}
                    
                    logs_df['treatment_name'] = logs_df['treatment_id'].map(treatment_map)
                    
                    # Create visualizations
                    if len(logs_df) > 0:
                        # Effectiveness comparison
                        fig1 = px.bar(
                            logs_df,
                            x='treatment_name',
                            y='effectiveness_rating',
                            labels={'treatment_name': 'Treatment', 'effectiveness_rating': 'Effectiveness Rating (1-5)'},
                            title='Treatment Effectiveness Comparison',
                            color='effectiveness_rating',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Timeline of treatments
                        if 'start_date' in logs_df.columns:
                            logs_df['start_date'] = pd.to_datetime(logs_df['start_date'])
                            
                            # If end_date is null, use today
                            logs_df['end_date'] = pd.to_datetime(logs_df['end_date'])
                            logs_df.loc[logs_df['end_date'].isnull(), 'end_date'] = pd.to_datetime('today')
                            
                            # Calculate duration
                            logs_df['duration_days'] = (logs_df['end_date'] - logs_df['start_date']).dt.days
                            
                            # Sort by start date
                            logs_df = logs_df.sort_values('start_date')
                            
                            # Create Gantt chart
                            fig2 = px.timeline(
                                logs_df,
                                x_start='start_date',
                                x_end='end_date',
                                y='treatment_name',
                                color='effectiveness_rating',
                                labels={'treatment_name': 'Treatment'},
                                title='Treatment Timeline',
                                color_continuous_scale=px.colors.sequential.Viridis
                            )
                            
                            # Update layout
                            fig2.update_yaxes(autorange="reversed")
                            
                            st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.subheader("Log New Treatment")
                    
                    # Fetch available treatments
                    treatments = fetch_treatments()
                    
                    if treatments:
                        with st.form("new_treatment_log_form"):
                            # Select treatment
                            treatment_id = st.selectbox(
                                "Select Treatment",
                                options=[t["id"] for t in treatments],
                                format_func=lambda x: next((t["name"] for t in treatments if t["id"] == x), x)
                            )
                            
                            # Treatment details
                            start_date = st.date_input("Start Date", value=datetime.now().date())
                            end_date = st.date_input("End Date (leave blank for ongoing)", value=None)
                            
                            dosage = st.text_input("Dosage (e.g., 10mg twice daily)")
                            frequency = st.text_input("Frequency (e.g., daily, weekly)")
                            
                            effectiveness_rating = st.slider(
                                "Effectiveness Rating",
                                min_value=1,
                                max_value=5,
                                value=3,
                                help="1 = No improvement, 5 = Cured"
                            )
                            
                            side_effects = st.text_area("Side Effects Experienced")
                            notes = st.text_area("Notes")
                            
                            is_active = st.checkbox("Treatment is Active/Ongoing", value=True)
                            
                            submit_button = st.form_submit_button("Log Treatment")
                            
                            if submit_button:
                                # Create treatment log entry
                                log_data = {
                                    "patient_id": patient['id'],
                                    "treatment_id": treatment_id,
                                    "start_date": start_date.isoformat(),
                                    "end_date": end_date.isoformat() if end_date else None,
                                    "dosage": dosage,
                                    "frequency": frequency,
                                    "effectiveness_rating": effectiveness_rating,
                                    "side_effects_experienced": side_effects,
                                    "notes": notes,
                                    "is_active": is_active
                                }
                                
                                # Create log
                                new_log = create_treatment_log(log_data)
                                
                                if new_log:
                                    st.success(f"Treatment logged successfully!")
                                else:
                                    st.error("Failed to log treatment. Please try again.")
                    else:
                        st.info("No treatments available. Add treatments first.")
            else:
                st.info("No treatment history found for this patient.")
                
                # Log New Treatment section
                st.subheader("Log New Treatment")
                
                # Fetch available treatments
                treatments = fetch_treatments()
                
                if treatments:
                    with st.form("new_treatment_log_form"):
                        # Select treatment
                        treatment_id = st.selectbox(
                            "Select Treatment",
                            options=[t["id"] for t in treatments],
                            format_func=lambda x: next((t["name"] for t in treatments if t["id"] == x), x)
                        )
                        
                        # Treatment details
                        start_date = st.date_input("Start Date", value=datetime.now().date())
                        end_date = st.date_input("End Date (leave blank for ongoing)", value=None)
                        
                        dosage = st.text_input("Dosage (e.g., 10mg twice daily)")
                        frequency = st.text_input("Frequency (e.g., daily, weekly)")
                        
                        effectiveness_rating = st.slider(
                            "Effectiveness Rating",
                            min_value=1,
                            max_value=5,
                            value=3,
                            help="1 = No improvement, 5 = Cured"
                        )
                        
                        side_effects = st.text_area("Side Effects Experienced")
                        notes = st.text_area("Notes")
                        
                        is_active = st.checkbox("Treatment is Active/Ongoing", value=True)
                        
                        submit_button = st.form_submit_button("Log Treatment")
                        
                        if submit_button:
                            # Create treatment log entry
                            log_data = {
                                "patient_id": patient['id'],
                                "treatment_id": treatment_id,
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat() if end_date else None,
                                "dosage": dosage,
                                "frequency": frequency,
                                "effectiveness_rating": effectiveness_rating,
                                "side_effects_experienced": side_effects,
                                "notes": notes,
                                "is_active": is_active
                            }
                            
                            # Create log
                            new_log = create_treatment_log(log_data)
                            
                            if new_log:
                                st.success(f"Treatment logged successfully!")
                            else:
                                st.error("Failed to log treatment. Please try again.")
                else:
                    st.info("No treatments available. Add treatments first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("FixImmune v0.1.1")
st.sidebar.caption("© 2023 FixImmune - An MVP Autoimmune Treatment Recommendation System")

# Main content footer
st.markdown("---")
st.caption("This application is for demonstration purposes only. Always consult with healthcare professionals for medical advice.") 