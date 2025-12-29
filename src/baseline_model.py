from sklearn.linear_model import LogisticRegression

def logistic_regression_model():
    """
    Interpretable baseline model.
    """
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
