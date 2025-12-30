import os
import joblib
import pandas as pd
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_fraud
from src.split import stratified_split
from src.ensemble_model import random_forest_model
from src.baseline_model import logistic_regression_model
from src.training_pipeline import train_and_evaluate

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_fraud_model(use_tuning=False, model_type="random_forest", search_method="random", n_iter=50):
    """
    Train fraud detection model with optional hyperparameter tuning.
    
    Args:
        use_tuning: Whether to use hyperparameter tuning
        model_type: Type of model ('logistic_regression' or 'random_forest')
        search_method: Search method ('grid' or 'random')
        n_iter: Number of iterations for randomized search
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv("data/processed/fraud_final.csv")
    df = preprocess_fraud(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = stratified_split(df, target="class")

    # Select base model
    if model_type == "random_forest":
        model = random_forest_model()
        model_file = f"{MODEL_DIR}/fraud_rf.pkl"
    elif model_type == "logistic_regression":
        model = logistic_regression_model()
        model_file = f"{MODEL_DIR}/fraud_lr.pkl"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train with optional hyperparameter tuning
    print(f"Training {model_type} model...")
    trained_model, metrics = train_and_evaluate(
        X_train, y_train, X_test, y_test, model,
        tune_hyperparameters=use_tuning,
        model_type=model_type,
        search_method=search_method,
        n_iter=n_iter,
        verbose=1
    )

    # Save model and features
    joblib.dump(trained_model, model_file)
    joblib.dump(X_train.columns.tolist(), f"{MODEL_DIR}/fraud_features.pkl")

    print(f"\n✅ {model_type} model trained and saved to {model_file}")
    print(f"📊 Test Metrics:")
    print(f"   F1 Score: {metrics['F1']:.4f}")
    print(f"   PR-AUC: {metrics['PR_AUC']:.4f}")
    print(f"   Confusion Matrix:\n{metrics['Confusion_Matrix']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Model type to train"
    )
    parser.add_argument(
        "--search",
        type=str,
        default="random",
        choices=["grid", "random"],
        help="Hyperparameter search method"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of iterations for randomized search"
    )
    
    args = parser.parse_args()
    
    train_fraud_model(
        use_tuning=args.tune,
        model_type=args.model,
        search_method=args.search,
        n_iter=args.n_iter
    )

