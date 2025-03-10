"""
Exploratory Data Analysis (EDA) and data engineering for FixImmune ML pipeline.

This module provides functions for analyzing and preprocessing the datasets
used in the FixImmune ML pipeline.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

from .download import DEFAULT_DATA_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Output directories
EDA_OUTPUT_DIR = DEFAULT_DATA_DIR / "eda"
PROCESSED_DATA_DIR = DEFAULT_DATA_DIR / "processed"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load a dataset from a file path.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Loaded dataset as a pandas DataFrame
    """
    # Check if path is a file or directory
    if dataset_path.is_file():
        # Check file extension to determine loading method
        if dataset_path.suffix == ".csv":
            return pd.read_csv(dataset_path)
        elif dataset_path.suffix == ".parquet":
            return pd.read_parquet(dataset_path)
        elif dataset_path.suffix == ".json":
            return pd.read_json(dataset_path)
        elif dataset_path.suffix == ".xlsx" or dataset_path.suffix == ".xls":
            return pd.read_excel(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
    
    # If it's a directory, try to find a dataset file inside
    elif dataset_path.is_dir():
        # Look for CSV files first
        csv_files = list(dataset_path.glob("*.csv"))
        if csv_files:
            # Load the first CSV file
            return pd.read_csv(csv_files[0])
        
        # Look for other supported file formats
        parquet_files = list(dataset_path.glob("*.parquet"))
        if parquet_files:
            return pd.read_parquet(parquet_files[0])
        
        json_files = list(dataset_path.glob("*.json"))
        if json_files:
            return pd.read_json(json_files[0])
        
        excel_files = list(dataset_path.glob("*.xlsx")) + list(dataset_path.glob("*.xls"))
        if excel_files:
            return pd.read_excel(excel_files[0])
        
        raise ValueError(f"No supported dataset files found in {dataset_path}")
    
    else:
        raise ValueError(f"Dataset path does not exist: {dataset_path}")


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about a dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing information about the dataset
    """
    # Get basic dataset information
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
        "datetime_columns": list(df.select_dtypes(include=["datetime", "timedelta"]).columns),
    }
    
    # Add summary statistics for numeric columns
    info["numeric_summary"] = df.describe().to_dict()
    
    # Add value counts for categorical columns
    info["categorical_summary"] = {
        col: df[col].value_counts().to_dict()
        for col in info["categorical_columns"]
        if df[col].nunique() < 50  # Only for columns with fewer than 50 unique values
    }
    
    return info


def generate_eda_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str
) -> None:
    """
    Generate EDA visualizations for a dataset.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save visualizations
        dataset_name: Name of the dataset (used for filenames)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique directory for this dataset
    dataset_dir = output_dir / dataset_name
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. Plot distribution of missing values
    plt.figure(figsize=(12, 6))
    missing_values = df.isnull().sum()
    missing_percentage = missing_values / len(df) * 100
    
    # Filter to only include columns with missing values
    missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
    
    if not missing_percentage.empty:
        # Create bar plot of missing values
        plt.bar(missing_percentage.index, missing_percentage.values)
        plt.xlabel("Columns")
        plt.ylabel("Missing Values (%)")
        plt.title(f"Missing Values (%) - {dataset_name}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(dataset_dir / "missing_values.png")
        plt.close()
    
    # 2. Plot distribution of numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 0:
        for i, col in enumerate(numeric_cols):
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col} - {dataset_name}")
            plt.tight_layout()
            plt.savefig(dataset_dir / f"distribution_{col}.png")
            plt.close()
        
        # 3. Create correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title(f"Correlation Matrix - {dataset_name}")
            plt.tight_layout()
            plt.savefig(dataset_dir / "correlation_matrix.png")
            plt.close()
    
    # 4. Plot bar charts for categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    
    for i, col in enumerate(categorical_cols):
        if df[col].nunique() < 20:  # Only for columns with fewer than 20 unique values
            plt.figure(figsize=(10, 6))
            df[col].value_counts().sort_values(ascending=False).plot(kind="bar")
            plt.title(f"Value Counts for {col} - {dataset_name}")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(dataset_dir / f"value_counts_{col}.png")
            plt.close()


def perform_data_cleaning(
    df: pd.DataFrame,
    dataset_name: str
) -> pd.DataFrame:
    """
    Perform basic data cleaning on a dataset.
    
    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset (for logging)
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning dataset: {dataset_name}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = df.copy()
    
    # 1. Drop columns with too many missing values (e.g., > 50%)
    missing_percentage = df_cleaned.isnull().sum() / len(df_cleaned) * 100
    cols_to_drop = missing_percentage[missing_percentage > 50].index.tolist()
    
    if cols_to_drop:
        logger.info(f"Dropping columns with >50% missing values: {cols_to_drop}")
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    
    # 2. Handle missing values
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        # Fill missing numeric values with median
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
            logger.info(f"Filled missing values in '{col}' with median: {median_value}")
    
    categorical_cols = df_cleaned.select_dtypes(include=["object", "category"]).columns
    
    for col in categorical_cols:
        # Fill missing categorical values with mode or "Unknown"
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].mode().empty:
                df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                logger.info(f"Filled missing values in '{col}' with 'Unknown'")
            else:
                mode_value = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                logger.info(f"Filled missing values in '{col}' with mode: {mode_value}")
    
    # 3. Drop duplicate rows
    duplicate_count = df_cleaned.duplicated().sum()
    if duplicate_count > 0:
        logger.info(f"Dropping {duplicate_count} duplicate rows")
        df_cleaned = df_cleaned.drop_duplicates()
    
    # 4. Convert date columns to datetime
    # Look for columns that might be dates
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    for col in categorical_cols:
        # Check if the column contains date-like strings
        if df_cleaned[col].dtype == "object" and df_cleaned[col].notna().any():
            sample_value = df_cleaned[col].dropna().iloc[0]
            if isinstance(sample_value, str) and pd.to_datetime(sample_value, errors="coerce") is not pd.NaT:
                logger.info(f"Converting column '{col}' to datetime")
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors="coerce")
    
    # 5. Convert categorical columns with few unique values to category dtype
    for col in categorical_cols:
        if df_cleaned[col].nunique() < 50:  # Only for columns with fewer than 50 unique values
            df_cleaned[col] = df_cleaned[col].astype("category")
            logger.info(f"Converted column '{col}' to category dtype")
    
    logger.info(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_cleaned.shape}")
    return df_cleaned


def analyze_treatment_effectiveness_data(
    df: pd.DataFrame,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Analyze treatment effectiveness data for autoimmune conditions.
    
    Args:
        df: Input DataFrame containing treatment effectiveness data
        output_dir: Directory to save analysis results
        
    Returns:
        Processed DataFrame with treatment effectiveness metrics
    """
    if output_dir is None:
        output_dir = EDA_OUTPUT_DIR / "treatment_effectiveness"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Analyzing treatment effectiveness data")
    
    # Perform data cleaning
    df_clean = perform_data_cleaning(df, "treatment_effectiveness")
    
    # Identify relevant columns for treatment effectiveness analysis
    treatment_cols = [col for col in df_clean.columns if "treatment" in col.lower()]
    condition_cols = [col for col in df_clean.columns if any(cond in col.lower() for cond in [
        "condition", "disease", "autoimmune", "symptom", "diagnosis"
    ])]
    effectiveness_cols = [col for col in df_clean.columns if any(eff in col.lower() for cond in [
        "effectiveness", "efficacy", "improvement", "response", "outcome", "score", "rating"
    ])]
    
    logger.info(f"Identified columns - Treatments: {treatment_cols}, Conditions: {condition_cols}, Effectiveness: {effectiveness_cols}")
    
    # Group data by treatment and condition (if available)
    result_dfs = []
    
    if treatment_cols and effectiveness_cols:
        for treatment_col in treatment_cols:
            for effectiveness_col in effectiveness_cols:
                if condition_cols:
                    for condition_col in condition_cols:
                        # Group by treatment and condition
                        grouped_df = df_clean.groupby([treatment_col, condition_col])[effectiveness_col].agg([
                            "mean", "median", "std", "count"
                        ]).reset_index()
                        
                        # Rename columns for clarity
                        grouped_df = grouped_df.rename(columns={
                            treatment_col: "treatment",
                            condition_col: "condition",
                            "mean": f"{effectiveness_col}_mean",
                            "median": f"{effectiveness_col}_median",
                            "std": f"{effectiveness_col}_std",
                            "count": f"{effectiveness_col}_count"
                        })
                        
                        result_dfs.append(grouped_df)
                else:
                    # Group by treatment only
                    grouped_df = df_clean.groupby([treatment_col])[effectiveness_col].agg([
                        "mean", "median", "std", "count"
                    ]).reset_index()
                    
                    # Rename columns for clarity
                    grouped_df = grouped_df.rename(columns={
                        treatment_col: "treatment",
                        "mean": f"{effectiveness_col}_mean",
                        "median": f"{effectiveness_col}_median",
                        "std": f"{effectiveness_col}_std",
                        "count": f"{effectiveness_col}_count"
                    })
                    
                    result_dfs.append(grouped_df)
    
    # Combine results if there are multiple groupings
    if result_dfs:
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        # Save results to CSV
        result_df.to_csv(output_dir / "treatment_effectiveness_analysis.csv", index=False)
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        if "treatment" in result_df.columns and any("mean" in col for col in result_df.columns):
            mean_col = next(col for col in result_df.columns if "mean" in col)
            try:
                # Sort by mean effectiveness
                plot_df = result_df.sort_values(by=mean_col, ascending=False).head(20)
                sns.barplot(data=plot_df, x="treatment", y=mean_col)
                plt.title("Treatment Effectiveness Comparison")
                plt.xlabel("Treatment")
                plt.ylabel(mean_col)
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(output_dir / "treatment_effectiveness_comparison.png")
                plt.close()
            except Exception as e:
                logger.error(f"Error creating visualization: {e}")
        
        return result_df
    else:
        logger.warning("Could not identify appropriate columns for treatment effectiveness analysis")
        return df_clean


def analyze_autoimmune_data(
    df: pd.DataFrame,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Analyze autoimmune disease data.
    
    Args:
        df: Input DataFrame containing autoimmune disease data
        output_dir: Directory to save analysis results
        
    Returns:
        Processed DataFrame with autoimmune disease metrics
    """
    if output_dir is None:
        output_dir = EDA_OUTPUT_DIR / "autoimmune"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Analyzing autoimmune disease data")
    
    # Perform data cleaning
    df_clean = perform_data_cleaning(df, "autoimmune")
    
    # Generate EDA visualizations
    generate_eda_visualizations(df_clean, output_dir, "autoimmune")
    
    # Save cleaned DataFrame
    df_clean.to_csv(output_dir / "autoimmune_data_cleaned.csv", index=False)
    
    return df_clean


def analyze_clinical_trials_data(
    df: pd.DataFrame,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Analyze clinical trials data for autoimmune conditions.
    
    Args:
        df: Input DataFrame containing clinical trials data
        output_dir: Directory to save analysis results
        
    Returns:
        Processed DataFrame with clinical trials metrics
    """
    if output_dir is None:
        output_dir = EDA_OUTPUT_DIR / "clinical_trials"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Analyzing clinical trials data")
    
    # Perform data cleaning
    df_clean = perform_data_cleaning(df, "clinical_trials")
    
    # Generate EDA visualizations
    generate_eda_visualizations(df_clean, output_dir, "clinical_trials")
    
    # Save cleaned DataFrame
    df_clean.to_csv(output_dir / "clinical_trials_data_cleaned.csv", index=False)
    
    return df_clean


def run_full_eda_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = EDA_OUTPUT_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Run a full EDA pipeline on all datasets.
    
    Args:
        data_dir: Directory containing the datasets
        output_dir: Directory to save EDA results
        
    Returns:
        Dictionary mapping dataset names to processed DataFrames
    """
    logger.info("Running full EDA pipeline")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    try:
        # Find dataset directories
        dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        # Process each dataset
        result_dfs = {}
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            logger.info(f"Processing dataset: {dataset_name}")
            
            try:
                # Load dataset
                df = load_dataset(dataset_dir)
                
                # Get basic dataset info
                dataset_info = get_dataset_info(df)
                
                # Save dataset info as JSON
                with open(output_dir / f"{dataset_name}_info.json", "w") as f:
                    # Convert some values to strings for JSON serialization
                    for key, value in dataset_info.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(v, np.ndarray):
                                    dataset_info[key][k] = v.tolist()
                                elif isinstance(v, pd.Series):
                                    dataset_info[key][k] = v.to_dict()
                    
                    json.dump(dataset_info, f, indent=2, default=str)
                
                # Generate EDA visualizations
                generate_eda_visualizations(df, output_dir, dataset_name)
                
                # Process dataset based on its name
                if "autoimmune" in dataset_name.lower():
                    result_dfs[dataset_name] = analyze_autoimmune_data(df, output_dir / dataset_name)
                elif "clinical" in dataset_name.lower() or "trial" in dataset_name.lower():
                    result_dfs[dataset_name] = analyze_clinical_trials_data(df, output_dir / dataset_name)
                elif "treatment" in dataset_name.lower() or "effectiveness" in dataset_name.lower():
                    result_dfs[dataset_name] = analyze_treatment_effectiveness_data(df, output_dir / dataset_name)
                else:
                    # Generic data cleaning and analysis
                    cleaned_df = perform_data_cleaning(df, dataset_name)
                    result_dfs[dataset_name] = cleaned_df
                    
                    # Save cleaned DataFrame
                    cleaned_df.to_csv(PROCESSED_DATA_DIR / f"{dataset_name}_cleaned.csv", index=False)
            
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
        
        logger.info(f"EDA completed for {len(result_dfs)} datasets")
        return result_dfs
    
    except Exception as e:
        logger.error(f"Error in EDA pipeline: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run full EDA pipeline
    processed_dfs = run_full_eda_pipeline()
    
    # Print summary of processed datasets
    for name, df in processed_dfs.items():
        print(f"Processed dataset: {name}, Shape: {df.shape}") 