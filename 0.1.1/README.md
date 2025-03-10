# FixImmune - Personalized Autoimmune Treatment Recommendation System

FixImmune is an MVP application designed to provide personalized treatment recommendations for individuals with autoimmune conditions. The goal is to help patients identify the most effective treatments based on their specific disease characteristics.

## Features

- **Patient Profile Management**: Create and manage profiles with detailed health information
- **AI-Driven Treatment Recommendations**: Receive personalized treatment suggestions based on patient data
- **Treatment Tracking**: Log and monitor treatments and their effectiveness
- **Clinical Trial Integration**: Access to relevant clinical trials and experimental treatments
- **Data-Driven Insights**: Utilize real-world effectiveness data to inform treatment decisions
- **ML-Powered Recommendations**: Advanced machine learning models to predict treatment effectiveness

## Setup

### Prerequisites

- Python 3.11 (installed via Conda)
- All required packages listed in `requirements.txt`

### Installation

1. Create a Conda environment with Python 3.11:
   ```
   conda create -n fiximmune python=3.11
   conda activate fiximmune
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file with your specific configuration parameters.

### Running the Application

Start the FastAPI backend:
```
python -m src.main
```

Start the ML recommendation API service (in a separate terminal):
```
python -m src.ml.run_ml_api
```

Start the Streamlit web interface (in a separate terminal):
```
streamlit run src/streamlit_app.py
```

## Project Structure

- `src/`: Main application code
  - `models/`: Data models and schemas
  - `services/`: Business logic and external service integrations
  - `api/`: API endpoints and routing
  - `database/`: Database models and connection handling
  - `utils/`: Utility functions
  - `ml/`: Machine learning pipeline
    - `data/`: Data acquisition and preprocessing
    - `features/`: Feature engineering and selection
    - `training/`: Model training and hyperparameter optimization
    - `inference/`: Model inference and prediction
    - `api/`: ML model API service
- `data/`: Sample data and datasets
- `tests/`: Test files

## ML Pipeline

The FixImmune ML pipeline consists of the following components:

1. **Data Acquisition**: Download datasets from Kaggle and other sources.
2. **Exploratory Data Analysis**: Analyze the data to understand patterns and relationships.
3. **Feature Engineering**: Create, encode, and scale features for modeling.
4. **Model Training**: Train XGBoost models with hyperparameter optimization using Optuna.
5. **Model Evaluation**: Evaluate model performance on validation and test datasets.
6. **Inference API**: Serve model predictions through a RESTful API.

### Running the ML Pipeline

To run the complete ML pipeline:

```
# Download datasets
python -m src.ml.data.download

# Run exploratory data analysis
python -m src.ml.data.eda

# Prepare features
python -m src.ml.features.feature_engineering

# Train and evaluate model
python -m src.ml.training.train_model

# Start the ML API server
python -m src.ml.run_ml_api
```

## Version

Current version: 0.1.1 