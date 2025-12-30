"""
Hyperparameter tuning module using cross-validation for fraud detection models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    make_scorer
)
import warnings
from typing import Dict, Any, Optional, Union


def _pr_auc_scorer(y_true, y_proba):
    """
    Custom scorer for Precision-Recall AUC.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
    
    Returns:
        PR-AUC score
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return auc(recall, precision)
    except Exception as e:
        warnings.warn(f"PR-AUC calculation failed: {str(e)}")
        return 0.0


def _f1_scorer(y_true, y_pred):
    """
    Custom scorer for F1 score.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
    
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred)


# Create custom scorers
pr_auc_scorer = make_scorer(
    _pr_auc_scorer,
    needs_proba=True,
    greater_is_better=True
)

f1_scorer = make_scorer(
    _f1_scorer,
    needs_proba=False,
    greater_is_better=True
)


def get_cv_strategy(n_splits=5, shuffle=True, random_state=42):
    """
    Get stratified K-fold cross-validation strategy for imbalanced data.
    
    Args:
        n_splits: Number of folds
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
    
    Returns:
        StratifiedKFold object
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    scoring: str = "f1",
    search_method: str = "grid",
    n_iter: int = 50,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Tune hyperparameters for Logistic Regression using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Cross-validation strategy (default: 5-fold stratified)
        scoring: Scoring metric ('f1', 'pr_auc', or 'both')
        search_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        n_iter: Number of iterations for randomized search
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level
    
    Returns:
        Dictionary containing:
        - 'best_model': Best model with tuned hyperparameters
        - 'best_params': Best hyperparameters
        - 'best_score': Best cross-validation score
        - 'cv_results': Full cross-validation results
        - 'scoring_metric': Metric used for optimization
    """
    from sklearn.linear_model import LogisticRegression
    
    # Define hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'max_iter': [500, 1000, 2000],
        'class_weight': ['balanced', None]
    }
    
    # Adjust solver based on penalty
    # L1 and elasticnet require specific solvers
    param_grid_adjusted = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],  # Start with L2 for simplicity
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [500, 1000, 2000],
        'class_weight': ['balanced', None]
    }
    
    # Select scoring metric
    if scoring == "f1":
        scoring_metric = f1_scorer
    elif scoring == "pr_auc":
        scoring_metric = pr_auc_scorer
    elif scoring == "both":
        scoring_metric = {
            'f1': f1_scorer,
            'pr_auc': pr_auc_scorer
        }
        # Use F1 as primary for optimization
        refit = 'f1'
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}. Use 'f1', 'pr_auc', or 'both'")
    
    # Set up cross-validation
    if cv is None:
        cv = get_cv_strategy(n_splits=5, random_state=random_state)
    
    # Create base model
    base_model = LogisticRegression(random_state=random_state)
    
    # Perform hyperparameter search
    if search_method == "grid":
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid_adjusted,
            cv=cv,
            scoring=scoring_metric if scoring != "both" else f1_scorer,
            refit=True if scoring != "both" else refit,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    elif search_method == "random":
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_adjusted,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring_metric if scoring != "both" else f1_scorer,
            refit=True if scoring != "both" else refit,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=True
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}. Use 'grid' or 'random'")
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Collect results
    results = {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
        'scoring_metric': scoring if scoring != "both" else 'f1'
    }
    
    # If both metrics requested, calculate PR-AUC for best model
    if scoring == "both":
        results['best_pr_auc'] = search.cv_results_[
            f'mean_test_pr_auc' if 'mean_test_pr_auc' in search.cv_results_ else 'mean_test_score'
        ][search.best_index_]
    
    return results


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    scoring: str = "f1",
    search_method: str = "random",
    n_iter: int = 50,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Tune hyperparameters for Random Forest using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Cross-validation strategy (default: 5-fold stratified)
        scoring: Scoring metric ('f1', 'pr_auc', or 'both')
        search_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        n_iter: Number of iterations for randomized search
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level
    
    Returns:
        Dictionary containing:
        - 'best_model': Best model with tuned hyperparameters
        - 'best_params': Best hyperparameters
        - 'best_score': Best cross-validation score
        - 'cv_results': Full cross-validation results
        - 'scoring_metric': Metric used for optimization
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Define hyperparameter grid/distribution
    if search_method == "grid":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
    else:  # random search - use distributions
        param_grid = {
            'n_estimators': [100, 150, 200, 250, 300, 400, 500],
            'max_depth': [5, 8, 10, 12, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
    
    # Select scoring metric
    if scoring == "f1":
        scoring_metric = f1_scorer
    elif scoring == "pr_auc":
        scoring_metric = pr_auc_scorer
    elif scoring == "both":
        scoring_metric = {
            'f1': f1_scorer,
            'pr_auc': pr_auc_scorer
        }
        refit = 'f1'
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}. Use 'f1', 'pr_auc', or 'both'")
    
    # Set up cross-validation
    if cv is None:
        cv = get_cv_strategy(n_splits=5, random_state=random_state)
    
    # Create base model
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Perform hyperparameter search
    if search_method == "grid":
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring_metric if scoring != "both" else f1_scorer,
            refit=True if scoring != "both" else refit,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    elif search_method == "random":
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring_metric if scoring != "both" else f1_scorer,
            refit=True if scoring != "both" else refit,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=True
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}. Use 'grid' or 'random'")
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Collect results
    results = {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
        'scoring_metric': scoring if scoring != "both" else 'f1'
    }
    
    # If both metrics requested, calculate PR-AUC for best model
    if scoring == "both":
        results['best_pr_auc'] = search.cv_results_[
            f'mean_test_pr_auc' if 'mean_test_pr_auc' in search.cv_results_ else 'mean_test_score'
        ][search.best_index_]
    
    return results


def tune_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    scoring: str = "f1",
    search_method: str = "random",
    n_iter: int = 50,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Unified interface for hyperparameter tuning.
    
    Args:
        model_type: Type of model ('logistic_regression' or 'random_forest')
        X_train: Training features
        y_train: Training target
        cv: Cross-validation strategy
        scoring: Scoring metric ('f1', 'pr_auc', or 'both')
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for randomized search
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level
    
    Returns:
        Dictionary with tuning results
    """
    model_type = model_type.lower()
    
    if model_type in ['logistic_regression', 'lr', 'logistic']:
        return tune_logistic_regression(
            X_train, y_train, cv, scoring, search_method,
            n_iter, n_jobs, random_state, verbose
        )
    elif model_type in ['random_forest', 'rf', 'randomforest']:
        return tune_random_forest(
            X_train, y_train, cv, scoring, search_method,
            n_iter, n_jobs, random_state, verbose
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Use 'logistic_regression' or 'random_forest'"
        )


def evaluate_cv_performance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    metrics: list = ['f1', 'pr_auc'],
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Evaluate model performance using cross-validation.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        cv: Cross-validation strategy
        metrics: List of metrics to compute
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary with mean and std of each metric
    """
    if cv is None:
        cv = get_cv_strategy(n_splits=5)
    
    results = {}
    
    for metric in metrics:
        if metric == 'f1':
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=f1_scorer, n_jobs=n_jobs
            )
        elif metric == 'pr_auc':
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=pr_auc_scorer, n_jobs=n_jobs
            )
        else:
            warnings.warn(f"Unknown metric: {metric}. Skipping.")
            continue
        
        results[f'{metric}_mean'] = scores.mean()
        results[f'{metric}_std'] = scores.std()
        results[f'{metric}_scores'] = scores
    
    return results

