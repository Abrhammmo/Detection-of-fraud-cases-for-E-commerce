from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
from typing import Optional, Dict, Any
import warnings


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    tune_hyperparameters: bool = False,
    model_type: Optional[str] = None,
    cv_folds: int = 5,
    scoring: str = "f1",
    search_method: str = "random",
    n_iter: int = 50,
    random_state: int = 42,
    verbose: int = 0
):
    """
    Train and evaluate a model with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model: Model instance to train (or base model if tuning)
        tune_hyperparameters: Whether to perform hyperparameter tuning
        model_type: Type of model ('logistic_regression' or 'random_forest')
                    Required if tune_hyperparameters=True
        cv_folds: Number of cross-validation folds for tuning
        scoring: Scoring metric for tuning ('f1', 'pr_auc', or 'both')
        search_method: Search method ('grid' or 'random')
        n_iter: Number of iterations for randomized search
        random_state: Random seed
        verbose: Verbosity level
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Perform hyperparameter tuning if requested
    if tune_hyperparameters:
        if model_type is None:
            # Try to infer model type from model class
            model_class_name = model.__class__.__name__
            if 'LogisticRegression' in model_class_name:
                model_type = 'logistic_regression'
            elif 'RandomForest' in model_class_name:
                model_type = 'random_forest'
            else:
                raise ValueError(
                    f"Cannot infer model type from {model_class_name}. "
                    "Please specify model_type parameter."
                )
        
        try:
            from src.hyperparameter_tuning import tune_model
            
            if verbose > 0:
                print(f"Tuning hyperparameters for {model_type} using {search_method} search...")
            
            tuning_results = tune_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                cv=None,  # Will use default 5-fold stratified CV
                scoring=scoring,
                search_method=search_method,
                n_iter=n_iter,
                random_state=random_state,
                verbose=verbose
            )
            
            model = tuning_results['best_model']
            
            if verbose > 0:
                print(f"Best parameters: {tuning_results['best_params']}")
                print(f"Best CV score ({tuning_results['scoring_metric']}): {tuning_results['best_score']:.4f}")
        
        except ImportError:
            warnings.warn(
                "Hyperparameter tuning module not found. Training with default parameters."
            )
        except Exception as e:
            warnings.warn(
                f"Hyperparameter tuning failed: {str(e)}. "
                "Training with default parameters."
            )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    
    # PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
    except Exception as e:
        warnings.warn(f"PR-AUC calculation failed: {str(e)}")
        pr_auc = 0.0

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "F1": f1,
        "PR_AUC": pr_auc,
        "Confusion_Matrix": cm
    }

    return model, metrics
