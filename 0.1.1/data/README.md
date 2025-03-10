# FixImmune Data Directory

This directory stores data files used by the FixImmune application.

## Sample Data

Sample data for the application is primarily managed programmatically through the `src/utils/sample_data.py` module, which initializes the database with:

- Treatment options for various autoimmune conditions
- Clinical trials information
- Sample patient profiles
- Treatment logs and effectiveness data

## External Data Sources

In a production environment, this directory could contain:

- Cached clinical trial data from ClinicalTrials.gov
- Real-world effectiveness datasets
- Machine learning model data for treatment recommendations
- Exported backups of the application database

## Data Privacy

In a production environment, any patient data would need to be properly anonymized and protected according to relevant healthcare data privacy regulations (e.g., HIPAA in the United States).

## Adding Custom Datasets

To add custom datasets to the application:

1. Place data files in this directory
2. Create a data loader in the `src/utils` directory
3. Update the appropriate service to use the new data 