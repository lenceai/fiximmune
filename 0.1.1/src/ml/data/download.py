"""
Download and prepare datasets for the FixImmune ML pipeline.

This module handles downloading datasets from Kaggle or other sources
and preparing them for use in the ML pipeline.
"""

import os
import logging
import pandas as pd
from pathlib import Path
import subprocess
import zipfile
import json
from typing import Optional, List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "ml"
KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_CREDENTIALS_PATH = KAGGLE_CONFIG_DIR / "kaggle.json"


def setup_kaggle_credentials(api_key: str, username: str) -> None:
    """
    Set up Kaggle credentials for API access.
    
    Args:
        api_key: Kaggle API key
        username: Kaggle username
    """
    # Create .kaggle directory if it doesn't exist
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
    
    # Create kaggle.json with credentials
    credentials = {
        "username": username,
        "key": api_key
    }
    
    # Write credentials to file
    with open(KAGGLE_CREDENTIALS_PATH, "w") as f:
        json.dump(credentials, f)
    
    # Set appropriate permissions (read/write for user only)
    os.chmod(KAGGLE_CREDENTIALS_PATH, 0o600)
    
    logger.info(f"Kaggle credentials saved to {KAGGLE_CREDENTIALS_PATH}")


def download_dataset_from_kaggle(
    dataset_name: str,
    output_dir: Optional[Path] = None,
    unzip: bool = True
) -> Path:
    """
    Download a dataset from Kaggle using the Kaggle API.
    
    Args:
        dataset_name: Name of the dataset on Kaggle (e.g., "author/dataset-name")
        output_dir: Directory to save the dataset (default: data/ml)
        unzip: Whether to unzip the downloaded file
        
    Returns:
        Path to the downloaded dataset directory
    """
    # Ensure Kaggle is installed
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle package not installed. Please run 'pip install kaggle'")
        raise
    
    # Set output directory
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists to avoid re-downloading
    dataset_slug = dataset_name.split("/")[-1]
    dataset_dir = output_dir / dataset_slug
    
    if dataset_dir.exists():
        logger.info(f"Dataset '{dataset_name}' already exists at {dataset_dir}")
        return dataset_dir
    
    # Download dataset
    logger.info(f"Downloading dataset '{dataset_name}' to {output_dir}")
    try:
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=unzip
        )
        logger.info(f"Downloaded dataset '{dataset_name}' successfully")
    except Exception as e:
        logger.error(f"Error downloading dataset '{dataset_name}': {e}")
        raise
    
    # Get the path to the downloaded dataset
    if unzip:
        # Dataset is unzipped to a directory
        return output_dir / dataset_slug
    else:
        # Dataset is a zip file
        return output_dir / f"{dataset_slug}.zip"


def download_autoimmune_dataset(
    output_dir: Optional[Path] = None,
    dataset_name: str = "sachinsdate/autoimmune-diseases-sjogren-lupus-covid19",
    unzip: bool = True
) -> Path:
    """
    Download the autoimmune diseases dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset on Kaggle
        unzip: Whether to unzip the downloaded file
        
    Returns:
        Path to the downloaded dataset directory
    """
    logger.info(f"Downloading autoimmune disease dataset: {dataset_name}")
    
    # Download the dataset
    dataset_path = download_dataset_from_kaggle(
        dataset_name=dataset_name,
        output_dir=output_dir,
        unzip=unzip
    )
    
    logger.info(f"Autoimmune disease dataset downloaded to {dataset_path}")
    return dataset_path


def download_clinical_trials_dataset(
    output_dir: Optional[Path] = None,
    dataset_name: str = "arashnic/clinicaltrials",
    unzip: bool = True
) -> Path:
    """
    Download the clinical trials dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset on Kaggle
        unzip: Whether to unzip the downloaded file
        
    Returns:
        Path to the downloaded dataset directory
    """
    logger.info(f"Downloading clinical trials dataset: {dataset_name}")
    
    # Download the dataset
    dataset_path = download_dataset_from_kaggle(
        dataset_name=dataset_name,
        output_dir=output_dir,
        unzip=unzip
    )
    
    logger.info(f"Clinical trials dataset downloaded to {dataset_path}")
    return dataset_path


def download_treatment_effectiveness_dataset(
    output_dir: Optional[Path] = None,
    dataset_name: str = "flaredown/flaredown-autoimmune-symptom-tracker",
    unzip: bool = True
) -> Path:
    """
    Download the treatment effectiveness dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset on Kaggle
        unzip: Whether to unzip the downloaded file
        
    Returns:
        Path to the downloaded dataset directory
    """
    logger.info(f"Downloading treatment effectiveness dataset: {dataset_name}")
    
    # Download the dataset
    dataset_path = download_dataset_from_kaggle(
        dataset_name=dataset_name,
        output_dir=output_dir,
        unzip=unzip
    )
    
    logger.info(f"Treatment effectiveness dataset downloaded to {dataset_path}")
    return dataset_path


def download_all_datasets(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Download all datasets needed for the FixImmune ML pipeline.
    
    Args:
        output_dir: Directory to save the datasets
        
    Returns:
        Dictionary mapping dataset names to their paths
    """
    logger.info("Downloading all datasets for FixImmune ML pipeline")
    
    # Set output directory
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download datasets
    datasets = {
        "autoimmune": download_autoimmune_dataset(output_dir),
        "clinical_trials": download_clinical_trials_dataset(output_dir),
        "treatment_effectiveness": download_treatment_effectiveness_dataset(output_dir)
    }
    
    logger.info(f"Downloaded {len(datasets)} datasets")
    return datasets


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Define data directory
    data_dir = DEFAULT_DATA_DIR
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Download all datasets
    datasets = download_all_datasets(data_dir)
    
    # Print dataset paths
    for name, path in datasets.items():
        print(f"{name}: {path}") 