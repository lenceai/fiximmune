"""
Model training with XGBoost and hyperparameter optimization.

This module handles training machine learning models for predicting
treatment effectiveness and generating personalized recommendations.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from optuna import create_study
from optuna.visualization import plot_optimization_history, plot_param_importances

# Local imports
from ..data.download import DEFAULT_DATA_DIR
from ..features.feature_engineering import FEATURES_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Output directories
MODELS_DIR = DEFAULT_DATA_DIR / "models"


def load_split_datasets(
    features_dir: Path = FEATURES_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Load the train, validation, and test datasets.
    
    Args:
        features_dir: Directory containing feature data splits
        
    Returns:
        Dictionary of DataFrames for train, validation, and test sets
    """
    logger.info(f"Loading split datasets from {features_dir}")
    
    # Load each dataset
    datasets = {}
    for split_name in ["train", "validation", "test"]:
        file_path = features_dir / f"{split_name}_data.csv"
        
        if file_path.exists():
            datasets[split_name] = pd.read_csv(file_path)
            logger.info(f"Loaded {split_name} dataset: {datasets[split_name].shape}")
        else:
            logger.warning(f"{split_name} dataset not found at {file_path}")
    
    # Check if we have at least training data
    if "train" not in datasets:
        raise ValueError("Training dataset not found")
    
    return datasets


def load_feature_metadata(features_dir: Path = FEATURES_DIR) -> Dict[str, Any]:
    """
    Load feature metadata from the features directory.
    
    Args:
        features_dir: Directory containing feature metadata
        
    Returns:
        Dictionary containing feature metadata
    """
    metadata_path = features_dir / "feature_metadata.json"
    
    if not metadata_path.exists():
        raise ValueError(f"Feature metadata not found at {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded feature metadata: {metadata['problem_type']} problem with {metadata.get('n_selected_features', 'unknown')} features")
    
    return metadata


def train_xgboost_model(
    train_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    val_df: Optional[pd.DataFrame] = None,
    hyperparams: Optional[Dict[str, Any]] = None
) -> xgb.Booster:
    """
    Train an XGBoost model on the provided data.
    
    Args:
        train_df: Training DataFrame
        target_column: Target column name
        problem_type: Type of problem ("classification" or "regression")
        val_df: Validation DataFrame (optional)
        hyperparams: Hyperparameters for XGBoost (optional)
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    # Setup validation data if available
    eval_set = None
    if val_df is not None:
        X_val = val_df.drop(columns=[target_column])
        y_val = val_df[target_column]
        eval_set = [(X_train, y_train), (X_val, y_val)]
    
    # Determine XGBoost objective based on problem type
    if problem_type == "classification":
        # Check if binary or multi-class
        if len(y_train.unique()) <= 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
    else:  # regression
        objective = "reg:squarederror"
        eval_metric = "rmse"
    
    # Set default hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "seed": 42
        }
    
    # Create and configure XGBoost model
    model_params = {
        "objective": objective,
        "eval_metric": eval_metric,
        **hyperparams
    }
    
    # Initialize XGBoost model
    model = xgb.XGBClassifier(**model_params) if problem_type == "classification" else xgb.XGBRegressor(**model_params)
    
    # Train the model
    if eval_set:
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Log best iteration and score
        logger.info(f"Best iteration: {model.best_iteration}")
        logger.info(f"Best score: {model.best_score}")
    else:
        model.fit(X_train, y_train)
    
    logger.info("XGBoost model training complete")
    
    return model


def hyperparameter_optimization_grid(
    train_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    val_df: Optional[pd.DataFrame] = None,
    n_jobs: int = -1,
    cv: int = 3
) -> Tuple[Dict[str, Any], Any]:
    """
    Perform grid search hyperparameter optimization for XGBoost.
    
    Args:
        train_df: Training DataFrame
        target_column: Target column name
        problem_type: Type of problem ("classification" or "regression")
        val_df: Validation DataFrame (optional)
        n_jobs: Number of parallel jobs (-1 for all processors)
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best hyperparameters, best model)
    """
    logger.info("Starting grid search hyperparameter optimization")
    
    # Prepare data
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
    }
    
    # Determine XGBoost objective based on problem type
    if problem_type == "classification":
        # Check if binary or multi-class
        if len(y.unique()) <= 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        
        # Create base model
        base_model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            use_label_encoder=False,
            seed=42
        )
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Define scoring metric
        scoring = 'accuracy'
    
    else:  # regression
        objective = "reg:squarederror"
        eval_metric = "rmse"
        
        # Create base model
        base_model = xgb.XGBRegressor(
            objective=objective,
            eval_metric=eval_metric,
            seed=42
        )
        
        # Define cross-validation strategy
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Define scoring metric
        scoring = 'neg_mean_squared_error'
    
    # Create and run grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=2
    )
    
    # Start timing
    start_time = time.time()
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Grid search completed in {elapsed_time:.2f} seconds")
    
    # Log best parameters and score
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best {scoring} score: {grid_search.best_score_}")
    
    return grid_search.best_params_, grid_search.best_estimator_


def hyperparameter_optimization_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform Optuna-based hyperparameter optimization for XGBoost.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        target_column: Target column name
        problem_type: Type of problem ("classification" or "regression")
        n_trials: Number of optimization trials
        timeout: Timeout in seconds (optional)
        output_dir: Directory to save optimization results (optional)
    
    Returns:
        Dictionary of best hyperparameters
    """
    logger.info(f"Starting Optuna hyperparameter optimization with {n_trials} trials")
    
    # Prepare training data
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    # Prepare validation data
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    
    # Determine if classification or regression
    is_classification = problem_type == "classification"
    
    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 10, log=True),
        }
        
        # Determine XGBoost objective based on problem type
        if is_classification:
            # Check if binary or multi-class
            if len(y_train.unique()) <= 2:
                params["objective"] = "binary:logistic"
                params["eval_metric"] = "logloss"
            else:
                params["objective"] = "multi:softprob"
                params["eval_metric"] = "mlogloss"
                params["num_class"] = len(y_train.unique())
            
            # Initialize XGBoost model
            model = xgb.XGBClassifier(**params, use_label_encoder=False, seed=42)
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
            
            # Initialize XGBoost model
            model = xgb.XGBRegressor(**params, seed=42)
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate the model
        if is_classification:
            y_pred = model.predict(X_val)
            
            # For binary classification, use roc_auc
            if len(y_train.unique()) <= 2:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
            else:
                score = accuracy_score(y_val, y_pred)
        else:
            y_pred = model.predict(X_val)
            score = -mean_squared_error(y_val, y_pred)  # Negative because Optuna minimizes
        
        return score
    
    # Create Optuna study
    study = create_study(direction="maximize")
    
    # Start optimization
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed_time = time.time() - start_time
    
    # Log results
    logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best score: {study.best_value}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    # Save visualization if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(str(output_dir / "optimization_history.png"))
        
        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image(str(output_dir / "param_importances.png"))
    
    return study.best_params


def train_optimized_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    optimization_method: str = "optuna",
    n_trials: int = 30,
    output_dir: Optional[Path] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train an optimized XGBoost model using hyperparameter optimization.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        target_column: Target column name
        problem_type: Type of problem ("classification" or "regression")
        optimization_method: Method for optimization ("grid", "random", "optuna")
        n_trials: Number of optimization trials (for optuna)
        output_dir: Directory to save optimization results (optional)
        
    Returns:
        Tuple of (optimized model, best hyperparameters)
    """
    logger.info(f"Training optimized model using {optimization_method} optimization")
    
    if output_dir is None:
        output_dir = MODELS_DIR / "optimization_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform hyperparameter optimization
    if optimization_method == "grid":
        best_params, best_model = hyperparameter_optimization_grid(
            train_df, target_column, problem_type, val_df
        )
        return best_model, best_params
    
    elif optimization_method == "optuna":
        best_params = hyperparameter_optimization_optuna(
            train_df, val_df, target_column, problem_type,
            n_trials=n_trials, output_dir=output_dir
        )
        
        # Train a new model with the best parameters
        model = train_xgboost_model(
            train_df, target_column, problem_type, val_df, hyperparams=best_params
        )
        
        return model, best_params
    
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")


def evaluate_model(
    model: Any,
    test_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        target_column: Target column name
        problem_type: Type of problem ("classification" or "regression")
        output_dir: Directory to save evaluation results (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data")
    
    if output_dir is None:
        output_dir = MODELS_DIR / "evaluation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare test data
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate based on problem type
    if problem_type == "classification":
        # For classification, evaluate using classification metrics
        # Check if binary or multi-class
        is_binary = len(np.unique(y_test)) <= 2
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted" if not is_binary else "binary"),
            "recall": recall_score(y_test, y_pred, average="weighted" if not is_binary else "binary"),
            "f1": f1_score(y_test, y_pred, average="weighted" if not is_binary else "binary"),
        }
        
        # For binary classification, compute ROC AUC
        if is_binary:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            except:
                logger.warning("Could not compute ROC AUC score")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save classification report
        with open(output_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
    
    else:  # regression
        # For regression, evaluate using regression metrics
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.savefig(output_dir / "actual_vs_predicted.png")
        plt.close()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig(output_dir / "residuals.png")
        plt.close()
    
    # Calculate feature importance
    try:
        feature_importance = model.feature_importances_
        feature_names = X_test.columns
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)
        
        # Save feature importance
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.bar(importance_df["Feature"][:20], importance_df["Importance"][:20])
        plt.xticks(rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png")
        plt.close()
    
    except:
        logger.warning("Could not compute feature importance")
    
    # Save metrics to file
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    return metrics


def save_model(
    model: Any,
    hyperparams: Dict[str, Any],
    metrics: Dict[str, Any],
    feature_metadata: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save a trained model and its metadata.
    
    Args:
        model: Trained model
        hyperparams: Hyperparameters used for training
        metrics: Evaluation metrics
        feature_metadata: Feature metadata
        output_dir: Directory to save the model (optional)
        
    Returns:
        Path where the model was saved
    """
    if output_dir is None:
        output_dir = MODELS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create model directory
    model_dir = output_dir / f"model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # For XGBoost models, also save in XGBoost binary format
    try:
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            xgb_model_path = model_dir / "model.xgb"
            model.save_model(str(xgb_model_path))
    except:
        logger.warning("Could not save model in XGBoost binary format")
    
    # Save hyperparameters
    with open(model_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    # Save metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save feature metadata
    with open(model_dir / "feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f, indent=4)
    
    # Create a model info file
    model_info = {
        "timestamp": timestamp,
        "problem_type": feature_metadata.get("problem_type", "unknown"),
        "target_column": feature_metadata.get("target_column", "unknown"),
        "metrics": metrics,
        "hyperparameters": hyperparams
    }
    
    with open(model_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model saved at {model_dir}")
    
    return model_dir


def train_and_evaluate_model(
    optimization_method: str = "optuna",
    n_trials: int = 30,
    features_dir: Path = FEATURES_DIR,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Train, optimize, evaluate, and save a model in a single workflow.
    
    Args:
        optimization_method: Method for hyperparameter optimization
        n_trials: Number of optimization trials (for optuna)
        features_dir: Directory containing feature data
        output_dir: Directory to save model and results (optional)
        
    Returns:
        Dictionary with model paths and performance information
    """
    logger.info("Starting model training and evaluation workflow")
    
    if output_dir is None:
        output_dir = MODELS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    datasets = load_split_datasets(features_dir)
    
    # Load feature metadata
    feature_metadata = load_feature_metadata(features_dir)
    target_column = feature_metadata["target_column"]
    problem_type = feature_metadata["problem_type"]
    
    # Train optimized model
    model, best_hyperparams = train_optimized_model(
        datasets["train"],
        datasets["validation"],
        target_column,
        problem_type,
        optimization_method=optimization_method,
        n_trials=n_trials,
        output_dir=output_dir / "optimization_results"
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model,
        datasets["test"],
        target_column,
        problem_type,
        output_dir=output_dir / "evaluation_results"
    )
    
    # Save model and metadata
    model_dir = save_model(
        model,
        best_hyperparams,
        metrics,
        feature_metadata,
        output_dir=output_dir
    )
    
    # Return results
    result = {
        "model_dir": str(model_dir),
        "model_path": str(model_dir / "model.pkl"),
        "hyperparameters": best_hyperparams,
        "metrics": metrics,
        "feature_metadata": feature_metadata
    }
    
    logger.info("Model training and evaluation workflow completed")
    
    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the training and evaluation workflow
    results = train_and_evaluate_model(
        optimization_method="optuna",
        n_trials=20  # Use fewer trials for quicker execution
    )
    
    # Print summary of results
    logger.info(f"Model saved at: {results['model_dir']}")
    logger.info(f"Best hyperparameters: {results['hyperparameters']}")
    logger.info(f"Evaluation metrics: {results['metrics']}") 