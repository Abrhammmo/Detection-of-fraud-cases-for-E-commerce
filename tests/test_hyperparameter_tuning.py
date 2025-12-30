import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join("..")))

from src.hyperparameter_tuning import (
    tune_logistic_regression,
    tune_random_forest,
    tune_model,
    get_cv_strategy,
    evaluate_cv_performance
)
from src.baseline_model import logistic_regression_model
from src.ensemble_model import random_forest_model


def test_cv_strategy():
    """Test that CV strategy is created correctly."""
    cv = get_cv_strategy(n_splits=5, random_state=42)
    assert cv.n_splits == 5
    assert cv.shuffle == True


def test_tune_logistic_regression():
    """Test hyperparameter tuning for Logistic Regression."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Test with minimal iterations for speed
    results = tune_logistic_regression(
        X_train=X,
        y_train=y,
        search_method="random",
        n_iter=5,
        verbose=0,
        random_state=42
    )
    
    # Assertions
    assert 'best_model' in results
    assert 'best_params' in results
    assert 'best_score' in results
    assert 'cv_results' in results
    assert results['best_score'] >= 0
    assert results['best_score'] <= 1


def test_tune_random_forest():
    """Test hyperparameter tuning for Random Forest."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Test with minimal iterations for speed
    results = tune_random_forest(
        X_train=X,
        y_train=y,
        search_method="random",
        n_iter=5,
        verbose=0,
        random_state=42
    )
    
    # Assertions
    assert 'best_model' in results
    assert 'best_params' in results
    assert 'best_score' in results
    assert 'cv_results' in results
    assert results['best_score'] >= 0
    assert results['best_score'] <= 1


def test_tune_model_unified_interface():
    """Test the unified tuning interface."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Test Logistic Regression
    results_lr = tune_model(
        model_type="logistic_regression",
        X_train=X,
        y_train=y,
        search_method="random",
        n_iter=5,
        verbose=0
    )
    assert 'best_model' in results_lr
    
    # Test Random Forest
    results_rf = tune_model(
        model_type="random_forest",
        X_train=X,
        y_train=y,
        search_method="random",
        n_iter=5,
        verbose=0
    )
    assert 'best_model' in results_rf


def test_evaluate_cv_performance():
    """Test cross-validation performance evaluation."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Train a simple model
    model = logistic_regression_model()
    model.fit(X, y)
    
    # Evaluate CV performance
    results = evaluate_cv_performance(
        model=model,
        X=X,
        y=y,
        metrics=['f1', 'pr_auc'],
        n_jobs=1
    )
    
    # Assertions
    assert 'f1_mean' in results
    assert 'f1_std' in results
    assert 'pr_auc_mean' in results
    assert 'pr_auc_std' in results
    assert 0 <= results['f1_mean'] <= 1
    assert 0 <= results['pr_auc_mean'] <= 1

