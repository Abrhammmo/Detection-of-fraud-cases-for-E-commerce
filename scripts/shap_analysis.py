import joblib
import pandas as pd
import shap
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_fraud
from src.split import stratified_split

def load_shap_components():
    model = joblib.load("../models/fraud_rf.pkl")
    features = joblib.load("../models/fraud_features.pkl")

    df = pd.read_csv("../data/processed/fraud_final.csv")
    df = preprocess_fraud(df)

    X_train, X_test, y_train, y_test = stratified_split(df, target="class")

    # Ensure feature alignment
    X_test = X_test.reindex(columns=features, fill_value=0)

    # Sample to avoid memory issues
    X_test = X_test.sample(n=1000, random_state=42)
    y_test = y_test.loc[X_test.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 🔑 RETURN FIVE VALUES
    return model, explainer, shap_values, X_test, y_test
