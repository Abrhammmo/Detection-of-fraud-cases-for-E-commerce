import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join("..")))

from src.preprocessing import preprocess_fraud
from src.split import stratified_split
from src.baseline_model import logistic_regression_model
from src.training_pipeline import train_and_evaluate


def test_training_pipeline_runs():
    """
    Test that the training pipeline runs and returns valid metrics.
    """

    # Load small sample to keep test fast
    df = pd.read_csv("data/processed/fraud_final.csv").sample(500, random_state=42)

    # Preprocess
    df = preprocess_fraud(df)

    # Split
    X_train, X_test, y_train, y_test = stratified_split(df, target="class")

    # Train model
    model = logistic_regression_model()
    _, metrics = train_and_evaluate(
        X_train, y_train, X_test, y_test, model
    )

    # Assertions
    assert isinstance(metrics, dict)
    assert "F1" in metrics
    assert "PR_AUC" in metrics
    assert "Confusion_Matrix" in metrics

    assert 0 <= metrics["F1"] <= 1
    assert 0 <= metrics["PR_AUC"] <= 1
