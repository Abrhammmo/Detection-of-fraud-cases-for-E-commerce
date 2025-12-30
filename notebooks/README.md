This folder contains Jupyter notebooks for the Detection of Fraud Cases for E-commerce project. These notebooks demonstrate the end-to-end process of building, evaluating, and interpreting a machine learning model for fraud detection using transaction data.

## Overview
The notebooks leverage Python libraries such as scikit-learn, pandas, and SHAP to preprocess data, train models, and provide explainability. The primary focus is on binary classification to identify fraudulent transactions, with an emphasis on model interpretability.

## Contents
shap-explainability.ipynb: This notebook focuses on model explainability using SHAP (SHapley Additive exPlanations). It includes:
Loading pre-trained models and SHAP components.
Generating global SHAP summary plots to visualize feature importance.
Creating force plots for individual predictions (true positives, false positives, and false negatives).
Comparing SHAP-based feature importance with the model's built-in feature importances (e.g., from Random Forest).
Observations and interpretations on how features like device_id_freq and time_since_signup influence fraud predictions.
## Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required libraries (install via pip install -r requirements.txt from the project root):
    - numpy
    - pandas
    - scikit-learn
    - shap
    - matplotlib
    - seaborn (if used in other notebooks)
## Setup
1. Clone or navigate to the project repository.
2. Ensure the virtual environment is activated (e.g., .venv in the project root).
3. Run the notebooks in order: Start with data loading and preprocessing (if applicable), then model training, and finally explainability.
4. The scripts folder (relative to the project root) contains helper functions like load_shap_components() used in the notebooks.
## Usage
1. Open the notebook in Jupyter: jupyter notebook shap-explainability.ipynb
2. Execute cells sequentially to reproduce results.
3. Modify parameters (e.g., model thresholds) to experiment with different scenarios.
4. For SHAP plots, ensure shap.initjs() is called for interactive visualizations in the browser.
## Notes
The notebooks assume pre-trained models and data splits are available via shap_analysis.py. If not, run the main training pipeline first.
SHAP computations can be computationally intensive; use a subset of data for faster prototyping.
Visualizations are optimized for Jupyter environments; export to HTML if needed.