"""
Feature engineering, selection, encoding, and scaling for FixImmune ML pipeline.

This module handles the transformation of raw data into features suitable for
training machine learning models for treatment recommendations.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
from ..data.download import DEFAULT_DATA_DIR
PROCESSED_DATA_DIR = DEFAULT_DATA_DIR / "processed"
FEATURES_DIR = DEFAULT_DATA_DIR / "features"


def identify_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify types of features in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping feature types to lists of column names
    """
    logger.info("Identifying feature types")
    
    # Initialize feature type categories
    feature_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "binary": [],
        "text": [],
        "id": [],
        "target": []
    }
    
    # Identify each column type
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column is likely an ID
        if any(id_term in col_lower for id_term in ["id", "uuid", "identifier", "key"]) and df[col].nunique() > df.shape[0] * 0.9:
            feature_types["id"].append(col)
        
        # Check if column is likely a target variable for treatment effectiveness
        elif any(target_term in col_lower for target_term in [
            "effectiveness", "efficacy", "outcome", "response", "improvement", 
            "score", "rating", "success", "result"
        ]):
            feature_types["target"].append(col)
        
        # Check data type for other features
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if binary
            if df[col].nunique() <= 2:
                feature_types["binary"].append(col)
            else:
                feature_types["numeric"].append(col)
        
        elif pd.api.types.is_datetime64_dtype(df[col]):
            feature_types["datetime"].append(col)
        
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Check if it's potentially a text field
            if df[col].dtype == "object" and df[col].str.len().mean() > 50:
                feature_types["text"].append(col)
            # Check if binary categorical
            elif df[col].nunique() <= 2:
                feature_types["binary"].append(col)
            else:
                feature_types["categorical"].append(col)
    
    # Log the results
    for feature_type, columns in feature_types.items():
        logger.info(f"Identified {len(columns)} {feature_type} features")
    
    return feature_types


def engineer_datetime_features(
    df: pd.DataFrame,
    datetime_columns: List[str]
) -> pd.DataFrame:
    """
    Engineer features from datetime columns.
    
    Args:
        df: Input DataFrame
        datetime_columns: List of datetime column names
        
    Returns:
        DataFrame with engineered datetime features
    """
    logger.info("Engineering datetime features")
    
    # Create a copy of the input DataFrame
    df_result = df.copy()
    
    # Process each datetime column
    for col in datetime_columns:
        if col in df_result.columns:
            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_dtype(df_result[col]):
                    df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
                
                # Extract useful components
                df_result[f"{col}_year"] = df_result[col].dt.year
                df_result[f"{col}_month"] = df_result[col].dt.month
                df_result[f"{col}_day"] = df_result[col].dt.day
                df_result[f"{col}_dayofweek"] = df_result[col].dt.dayofweek
                df_result[f"{col}_quarter"] = df_result[col].dt.quarter
                
                # Calculate days since a reference date (e.g., start of dataset)
                min_date = df_result[col].min()
                if pd.notna(min_date):
                    df_result[f"{col}_days_since_min"] = (df_result[col] - min_date).dt.days
                
                logger.info(f"Created datetime features for column '{col}'")
            
            except Exception as e:
                logger.error(f"Error engineering datetime features for column '{col}': {e}")
    
    return df_result


def engineer_text_features(
    df: pd.DataFrame,
    text_columns: List[str]
) -> pd.DataFrame:
    """
    Engineer features from text columns.
    
    Args:
        df: Input DataFrame
        text_columns: List of text column names
        
    Returns:
        DataFrame with engineered text features
    """
    logger.info("Engineering text features")
    
    # Create a copy of the input DataFrame
    df_result = df.copy()
    
    # Process each text column
    for col in text_columns:
        if col in df_result.columns:
            try:
                # Convert to string if needed
                if df_result[col].dtype != "object":
                    df_result[col] = df_result[col].astype(str)
                
                # Create basic text features
                df_result[f"{col}_length"] = df_result[col].str.len()
                df_result[f"{col}_word_count"] = df_result[col].str.split().str.len()
                df_result[f"{col}_uppercase_ratio"] = df_result[col].apply(
                    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
                )
                
                logger.info(f"Created text features for column '{col}'")
            
            except Exception as e:
                logger.error(f"Error engineering text features for column '{col}': {e}")
    
    return df_result


def create_treatment_specific_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create features specific to treatment recommendation tasks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with treatment-specific features
    """
    logger.info("Creating treatment-specific features")
    
    # Create a copy of the input DataFrame
    df_result = df.copy()
    
    # Look for treatment and condition columns
    treatment_cols = [col for col in df_result.columns if "treatment" in col.lower()]
    condition_cols = [col for col in df_result.columns if any(term in col.lower() for term in [
        "condition", "disease", "diagnosis", "autoimmune"
    ])]
    symptom_cols = [col for col in df_result.columns if "symptom" in col.lower()]
    
    # Create treatment-condition interaction features
    if treatment_cols and condition_cols:
        for treatment_col in treatment_cols:
            for condition_col in condition_cols:
                # Create interaction feature if both columns are categorical
                if df_result[treatment_col].dtype == "object" and df_result[condition_col].dtype == "object":
                    df_result[f"{treatment_col}_{condition_col}_interaction"] = df_result[treatment_col] + "_" + df_result[condition_col]
                    logger.info(f"Created interaction feature: {treatment_col}_{condition_col}_interaction")
    
    # Create symptom severity indicators if available
    if symptom_cols:
        for symptom_col in symptom_cols:
            if pd.api.types.is_numeric_dtype(df_result[symptom_col]):
                # Create severity level indicators
                median_severity = df_result[symptom_col].median()
                df_result[f"{symptom_col}_high"] = (df_result[symptom_col] > median_severity).astype(int)
                logger.info(f"Created high severity indicator for: {symptom_col}")
    
    return df_result


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoding_method: str = "onehot",
    handle_unknown: str = "ignore",
    max_categories: int = 20
) -> Tuple[pd.DataFrame, Any]:
    """
    Encode categorical features using the specified method.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names
        encoding_method: Encoding method ("onehot", "label", "ordinal")
        handle_unknown: How to handle unknown categories ("ignore", "error")
        max_categories: Maximum number of categories to consider for one-hot encoding
        
    Returns:
        Tuple of (DataFrame with encoded features, encoder object)
    """
    logger.info(f"Encoding categorical features using {encoding_method} encoding")
    
    # Create a copy of the input DataFrame
    df_result = df.copy()
    
    # Filter only existing categorical columns
    valid_cat_cols = [col for col in categorical_columns if col in df_result.columns]
    
    if not valid_cat_cols:
        logger.warning("No valid categorical columns to encode")
        return df_result, None
    
    # Initialize encoder based on method
    if encoding_method == "onehot":
        encoder = OneHotEncoder(sparse=False, handle_unknown=handle_unknown)
        
        # Filter columns with too many categories
        filtered_cat_cols = []
        for col in valid_cat_cols:
            num_categories = df_result[col].nunique()
            if num_categories <= max_categories:
                filtered_cat_cols.append(col)
            else:
                logger.warning(f"Skipping column '{col}' for one-hot encoding (has {num_categories} categories)")
        
        # Proceed only if there are columns to encode
        if filtered_cat_cols:
            # Fit and transform the data
            encoded_array = encoder.fit_transform(df_result[filtered_cat_cols])
            
            # Get feature names
            try:
                feature_names = encoder.get_feature_names_out(filtered_cat_cols)
            except:
                # Fallback for older sklearn versions
                feature_names = [f"{col}_{cat}" for i, col in enumerate(filtered_cat_cols) 
                                for cat in encoder.categories_[i]]
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_result.index)
            
            # Drop original categorical columns and join encoded ones
            df_result = df_result.drop(columns=filtered_cat_cols)
            df_result = pd.concat([df_result, encoded_df], axis=1)
            
            logger.info(f"Created {len(feature_names)} one-hot encoded features from {len(filtered_cat_cols)} categorical columns")
        
    elif encoding_method == "label":
        encoder = {}
        
        # Encode each categorical column separately
        for col in valid_cat_cols:
            label_encoder = LabelEncoder()
            df_result[col] = label_encoder.fit_transform(df_result[col].astype(str))
            encoder[col] = label_encoder
            
            logger.info(f"Label encoded column '{col}' with {len(label_encoder.classes_)} categories")
    
    elif encoding_method == "ordinal":
        encoder = OrdinalEncoder(handle_unknown=handle_unknown)
        
        # Fit and transform all categorical columns
        df_result[valid_cat_cols] = encoder.fit_transform(df_result[valid_cat_cols])
        
        logger.info(f"Ordinal encoded {len(valid_cat_cols)} categorical columns")
    
    else:
        raise ValueError(f"Unsupported encoding method: {encoding_method}")
    
    return df_result, encoder


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    scaling_method: str = "standard"
) -> Tuple[pd.DataFrame, Any]:
    """
    Scale numeric features using the specified method.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
        scaling_method: Scaling method ("standard", "minmax", "robust")
        
    Returns:
        Tuple of (DataFrame with scaled features, scaler object)
    """
    logger.info(f"Scaling numeric features using {scaling_method} scaling")
    
    # Create a copy of the input DataFrame
    df_result = df.copy()
    
    # Filter only existing numeric columns
    valid_num_cols = [col for col in numeric_columns if col in df_result.columns]
    
    if not valid_num_cols:
        logger.warning("No valid numeric columns to scale")
        return df_result, None
    
    # Initialize scaler based on method
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")
    
    # Fit and transform numeric columns
    df_result[valid_num_cols] = scaler.fit_transform(df_result[valid_num_cols])
    
    logger.info(f"Scaled {len(valid_num_cols)} numeric columns")
    
    return df_result, scaler


def select_features(
    df: pd.DataFrame,
    target_column: str,
    n_features: int = None,
    method: str = "selectkbest",
    problem_type: str = "classification"
) -> Tuple[pd.DataFrame, Any]:
    """
    Select the most important features for the target variable.
    
    Args:
        df: Input DataFrame
        target_column: Target variable column name
        n_features: Number of features to select (default: None, selects all features)
        method: Feature selection method ("selectkbest", "rfe", "model_based", "pca")
        problem_type: Problem type ("classification" or "regression")
        
    Returns:
        Tuple of (DataFrame with selected features, selector object)
    """
    logger.info(f"Selecting features using {method} method")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Determine appropriate feature selection for problem type
    if problem_type == "classification":
        scoring_function = f_classif
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # regression
        scoring_function = f_regression
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Set default n_features if not provided
    if n_features is None:
        n_features = X.shape[1] // 2  # Select half of the features by default
    
    # Initialize selector based on method
    if method == "selectkbest":
        selector = SelectKBest(score_func=scoring_function, k=n_features)
    elif method == "rfe":
        selector = RFE(estimator=base_model, n_features_to_select=n_features)
    elif method == "model_based":
        selector = SelectFromModel(estimator=base_model, max_features=n_features)
    elif method == "pca":
        selector = PCA(n_components=n_features)
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")
    
    # Fit and transform the data
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    if method == "selectkbest":
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
    elif method == "rfe" or method == "model_based":
        selected_features = X.columns[selector.get_support()].tolist()
    elif method == "pca":
        # For PCA, we don't have direct feature names since we've created new components
        selected_features = [f"PC{i+1}" for i in range(n_features)]
    
    # Create DataFrame with selected features
    if method == "pca":
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    else:
        X_selected_df = X[selected_features]
    
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Create a full DataFrame with both X and y
    result_df = pd.concat([X_selected_df, y], axis=1)
    
    return result_df, selector


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Target variable column name
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        stratify: Whether to perform stratified splitting (for classification)
        
    Returns:
        Dictionary containing train, validation, and test DataFrames
    """
    logger.info("Splitting data into train, validation, and test sets")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Determine if stratification should be used
    stratify_y = None
    if stratify:
        # Check if target is categorical or binary
        if pd.api.types.is_categorical_dtype(y) or len(y.unique()) <= 10:
            stratify_y = y
            logger.info("Using stratified splitting")
    
    # First split: separate training data from test data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    
    # Update stratify_y for the second split
    if stratify_y is not None:
        stratify_y = y_train_val
    
    # Second split: separate training data from validation data
    # Adjust validation size to account for the first split
    adjusted_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, 
        random_state=random_state, stratify=stratify_y
    )
    
    # Combine features and targets back into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    logger.info(f"Data split - Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")
    
    # Return splits as a dictionary
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }


def create_feature_pipeline(
    categorical_columns: List[str],
    numeric_columns: List[str],
    datetime_columns: List[str] = None,
    text_columns: List[str] = None,
    categorical_encoding: str = "onehot",
    numeric_scaling: str = "standard",
    handle_unknown: str = "ignore",
    max_categories: int = 20
) -> Pipeline:
    """
    Create a feature transformation pipeline.
    
    Args:
        categorical_columns: List of categorical column names
        numeric_columns: List of numeric column names
        datetime_columns: List of datetime column names
        text_columns: List of text column names
        categorical_encoding: Encoding method for categorical features
        numeric_scaling: Scaling method for numeric features
        handle_unknown: How to handle unknown categories
        max_categories: Maximum number of categories for one-hot encoding
        
    Returns:
        Scikit-learn pipeline for feature transformation
    """
    logger.info("Creating feature transformation pipeline")
    
    # Initialize transformers list
    transformers = []
    
    # Add categorical encoder
    if categorical_columns:
        if categorical_encoding == "onehot":
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown=handle_unknown, max_categories=max_categories))
            ])
        elif categorical_encoding == "ordinal":
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OrdinalEncoder(handle_unknown=handle_unknown))
            ])
        else:
            raise ValueError(f"Unsupported categorical encoding: {categorical_encoding}")
        
        transformers.append(("cat", categorical_transformer, categorical_columns))
    
    # Add numeric scaler
    if numeric_columns:
        if numeric_scaling == "standard":
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
        elif numeric_scaling == "minmax":
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler())
            ])
        elif numeric_scaling == "robust":
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ])
        else:
            raise ValueError(f"Unsupported numeric scaling: {numeric_scaling}")
        
        transformers.append(("num", numeric_transformer, numeric_columns))
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"  # Drop columns not explicitly specified
    )
    
    # Create and return the pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    
    logger.info(f"Created feature pipeline with {len(transformers)} transformers")
    
    return pipeline


def prepare_treatment_recommendation_features(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    n_features: int = None,
    feature_selection_method: str = "selectkbest"
) -> Dict[str, Any]:
    """
    Prepare features for training a treatment recommendation model.
    
    Args:
        df: Input DataFrame
        target_column: Target variable column name
        output_dir: Directory to save processed data
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        n_features: Number of features to select
        feature_selection_method: Feature selection method
        
    Returns:
        Dictionary containing processed datasets and feature information
    """
    if output_dir is None:
        output_dir = FEATURES_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Preparing features for treatment recommendation model")
    
    # Identify feature types
    feature_types = identify_feature_types(df)
    
    # Engineer features
    df_engineered = engineer_datetime_features(df, feature_types.get("datetime", []))
    df_engineered = engineer_text_features(df_engineered, feature_types.get("text", []))
    df_engineered = create_treatment_specific_features(df_engineered)
    
    # Update feature types after engineering
    feature_types = identify_feature_types(df_engineered)
    
    # Encode categorical features
    df_encoded, categorical_encoder = encode_categorical_features(
        df_engineered, 
        feature_types.get("categorical", []) + feature_types.get("binary", []),
        encoding_method="onehot"
    )
    
    # Scale numeric features
    df_scaled, numeric_scaler = scale_numeric_features(
        df_encoded,
        feature_types.get("numeric", []),
        scaling_method="standard"
    )
    
    # Determine problem type
    if pd.api.types.is_categorical_dtype(df_scaled[target_column]) or df_scaled[target_column].nunique() <= 5:
        problem_type = "classification"
    else:
        problem_type = "regression"
    
    logger.info(f"Detected problem type: {problem_type}")
    
    # Select features
    if n_features is not None:
        df_selected, feature_selector = select_features(
            df_scaled,
            target_column=target_column,
            n_features=n_features,
            method=feature_selection_method,
            problem_type=problem_type
        )
    else:
        df_selected = df_scaled
        feature_selector = None
    
    # Split the data
    data_splits = split_data(
        df_selected,
        target_column=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=(problem_type == "classification")
    )
    
    # Save processed datasets
    for split_name, split_df in data_splits.items():
        split_df.to_csv(output_dir / f"{split_name}_data.csv", index=False)
        logger.info(f"Saved {split_name} dataset: {split_df.shape}")
    
    # Create a feature metadata file
    feature_metadata = {
        "feature_types": {k: v for k, v in feature_types.items() if v},
        "target_column": target_column,
        "problem_type": problem_type,
        "original_shape": df.shape,
        "processed_shape": df_selected.shape,
        "n_selected_features": df_selected.shape[1] - 1,  # Subtract 1 for target column
        "train_shape": data_splits["train"].shape,
        "validation_shape": data_splits["validation"].shape,
        "test_shape": data_splits["test"].shape
    }
    
    # Save feature metadata
    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f, indent=2, default=str)
    
    # Save encoders and scalers
    if categorical_encoder is not None:
        with open(output_dir / "categorical_encoder.pkl", "wb") as f:
            pickle.dump(categorical_encoder, f)
    
    if numeric_scaler is not None:
        with open(output_dir / "numeric_scaler.pkl", "wb") as f:
            pickle.dump(numeric_scaler, f)
    
    if feature_selector is not None:
        with open(output_dir / "feature_selector.pkl", "wb") as f:
            pickle.dump(feature_selector, f)
    
    logger.info("Feature preparation complete")
    
    # Return processed data and metadata
    return {
        "data_splits": data_splits,
        "feature_metadata": feature_metadata,
        "categorical_encoder": categorical_encoder,
        "numeric_scaler": numeric_scaler,
        "feature_selector": feature_selector
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage: Load a processed dataset and prepare features
    try:
        # Find processed datasets
        processed_files = list(PROCESSED_DATA_DIR.glob("*.csv"))
        
        if processed_files:
            # Use the first processed dataset
            dataset_path = processed_files[0]
            logger.info(f"Using dataset: {dataset_path}")
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Identify potential target columns (effectiveness-related)
            potential_targets = [col for col in df.columns if any(term in col.lower() for term in [
                "effectiveness", "efficacy", "outcome", "response", "rating", "score"
            ])]
            
            if potential_targets:
                target_column = potential_targets[0]
                logger.info(f"Using target column: {target_column}")
                
                # Prepare features
                result = prepare_treatment_recommendation_features(
                    df=df,
                    target_column=target_column,
                    n_features=20  # Select top 20 features
                )
                
                logger.info("Feature preparation complete")
            else:
                logger.error("No suitable target column found")
        else:
            logger.error("No processed datasets found")
    
    except Exception as e:
        logger.error(f"Error in feature preparation: {e}")
        raise 