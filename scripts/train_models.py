import os
import joblib
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_fraud
from src.split import stratified_split
from src.ensemble_model import random_forest_model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_fraud_model():
    df = pd.read_csv("data/processed/fraud_final.csv")
    df = preprocess_fraud(df)

    X_train, X_test, y_train, y_test = stratified_split(df, target="class")

    model = random_forest_model()
    model.fit(X_train, y_train)

    joblib.dump(model, f"{MODEL_DIR}/fraud_rf.pkl")
    joblib.dump(X_train.columns.tolist(), f"{MODEL_DIR}/fraud_features.pkl")

    print("✅ Fraud model trained and saved")

if __name__ == "__main__":
    train_fraud_model()

