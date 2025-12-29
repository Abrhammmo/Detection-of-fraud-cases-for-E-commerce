from sklearn.ensemble import RandomForestClassifier

def random_forest_model():
    """
    Ensemble model for fraud detection.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
