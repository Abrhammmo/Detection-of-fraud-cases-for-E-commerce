from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix

def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

    # F1 Score
    f1 = f1_score(y_test, y_pred)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "F1": f1,
        "PR_AUC": pr_auc,
        "Confusion_Matrix": cm
    }

    return model, metrics
