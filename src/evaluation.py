from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix
)

def evaluate(model, X_test, y_test):
    """
    Evaluate model using imbalanced-aware metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "F1": f1_score(y_test, y_pred),
        "AUC_PR": average_precision_score(y_test, y_prob),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred)
    }
