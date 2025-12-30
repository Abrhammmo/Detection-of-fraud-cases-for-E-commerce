# Project: Multi-Platform Fraud Detection System

**Developed for Adey Innovations Inc.**

## 📌 Overview

At **Adey Innovations Inc.**, securing financial transactions is at the core of our mission. This project delivers a robust, machine-learning-driven fraud detection system designed to protect both e-commerce platforms and banking institutions.

By integrating geolocation analysis, transaction velocity patterns, and advanced ensemble modeling, we provide a solution that minimizes financial loss (False Negatives) while preserving a seamless user experience (minimizing False Positives).

---

## 🏢 Business Objective

The financial technology sector faces a constant battle against evolving fraud tactics. Our goal is to:

* **Enhance Security:** Detect sophisticated fraud patterns in real-time.
* **Balance UX & Risk:** Optimize the trade-off between strict security and customer friction.
* **Build Trust:** Provide transparent, explainable AI insights to stakeholders and financial partners.

---

## 🛠️ Technical Workflow

### 1. Data Analysis & Geolocation Intelligence

We process two distinct streams: e-commerce logs and bank credit transactions.

* **IP Intelligence:** We map transaction IP addresses to physical locations to identify high-risk geographic anomalies.
* **Feature Engineering:** Beyond raw data, we calculate **Transaction Velocity** (the speed of repeated purchases) and **Account Aging** (time elapsed since signup) to flag bot-like behavior.

### 2. Predictive Modeling & Evaluation

Since fraudulent transactions are rare compared to legitimate ones, we utilize specialized techniques to prevent the model from being "blinded" by the majority class.

* **Handling Imbalance:** We utilize **SMOTE** (Synthetic Minority Over-sampling Technique) to create a balanced learning environment.
* **Ensemble Power:** We deploy **XGBoost** and **Random Forest** models, which are superior at capturing the complex, non-linear relationships typical of financial crimes.
* **Precision-Recall Focus:** We evaluate success using the **Precision-Recall Curve (AUC-PR)** and **F1-Score**, ensuring we prioritize the detection of fraud without overwhelming the system with false alarms.

### 3. Explainability & Actionable Insights

A "black box" model is not enough for the fintech industry. We use **SHAP (SHapley Additive exPlanations)** to break down why a specific transaction was flagged.

* **Global Drivers:** Identify what factors most consistently indicate fraud across the company.
* **Individual Case Studies:** Analyze "False Positives" to refine rules and reduce customer annoyance.
* **Business Rules:** Translate model data into simple logic, such as: *"Flag all accounts where the purchase occurs within 10 minutes of creation from a high-risk IP range."*

---

## 📈 Impact

* **Reduced Financial Leakage:** Early detection of "burn accounts" prevents immediate loss.
* **Operational Efficiency:** Automated screening allows security teams to focus on high-probability cases.
* **Scalability:** The framework is designed to adapt to new datasets as Adey Innovations expands its financial reach.


## 🧠 Task 2 — Model Building, Training & Evaluation

### Objective

Design, train, and evaluate **robust classification models** capable of detecting fraudulent transactions in **highly imbalanced datasets**, while ensuring **reproducibility, modularity, and fair model comparison**.

This task focuses on **modeling rigor**, **evaluation correctness**, and **engineering best practices**.

---

## 📊 1. Data Preparation

### Dataset Separation

Two datasets are modeled **independently** due to different feature spaces and business contexts:

* **E-commerce transactions**

  * Target: `class`
  * Behavioral + geolocation features

* **Credit card transactions**

  * Target: `Class`
  * PCA-transformed numerical features (V1–V28)

> The datasets are **not merged** to avoid feature leakage and semantic mismatch.

---

### Stratified Train–Test Split

To preserve the extreme fraud imbalance:

* **Stratified splitting** ensures class distribution consistency
* Prevents misleading performance inflation

```text
Train/Test split: 80% / 20%
Stratification key: fraud label
```

---

## 🧩 2. Modular Training Pipeline

All modeling logic is implemented in the **`src/` directory** to ensure:

* Reusability
* Testability
* CI/CD compatibility
* Clean notebook execution

### Core Pipeline Components

| Module                 | Responsibility                 |
| ---------------------- | ------------------------------ |
| `preprocessing.py`     | Feature preparation & encoding |
| `split.py`             | Stratified train-test split    |
| `baseline_model.py`    | Logistic Regression definition |
| `ensemble_model.py`    | Random Forest model            |
| `training_pipeline.py` | Unified training & evaluation  |

---

## 📉 3. Baseline Model — Logistic Regression

### Why Logistic Regression?

* Highly interpretable
* Establishes a **performance floor**
* Ideal for business explanation and risk justification

### Configuration

* Class weighting to account for imbalance
* Regularization to reduce overfitting

### Metrics Used

* **F1-Score** → Balance between Precision & Recall
* **PR-AUC** → Preferred over ROC-AUC for rare events
* **Confusion Matrix** → Operational error analysis

---

## 🌲 4. Ensemble Model — Random Forest

### Why Random Forest?

* Handles non-linear interactions
* Robust to noise
* Strong performance on tabular fraud data

### Tuned Hyperparameters

* `n_estimators`
* `max_depth`
* `min_samples_split`

### Strengths

* Captures behavioral patterns
* Learns complex fraud signatures
* Resistant to overfitting compared to single trees

---

## ⚖️ 5. Evaluation Strategy (Imbalanced Learning)

Fraud detection is **not an accuracy problem**.

### Chosen Metrics

| Metric               | Justification                               |
| -------------------- | ------------------------------------------- |
| **F1-Score**         | Penalizes false positives & false negatives |
| **PR-AUC**           | Focuses on minority (fraud) class           |
| **Confusion Matrix** | Business-impact clarity                     |

> ROC-AUC is avoided as it can be misleading under extreme imbalance.

---

## 🔍 6. Cross-Dataset Model Comparison

Models are trained and evaluated **separately** on:

* Fraud (E-commerce) dataset
* Credit Card dataset

### Observed Patterns

* Random Forest significantly outperforms Logistic Regression
* Credit card data benefits from PCA-engineered features
* E-commerce data benefits from behavioral + geolocation features

---

## 🧪 7. Validation & Testing

To ensure robustness:

* **Unit tests** validate pipeline execution
* Metrics are sanity-checked (0 ≤ score ≤ 1)
* Training is deterministic via fixed random seeds

```bash
pytest
```

---

## 🏆 8. Model Selection & Justification

| Model               | Selection Reason                         |
| ------------------- | ---------------------------------------- |
| Logistic Regression | Interpretability & baseline benchmarking |
| Random Forest       | Best overall fraud detection performance |

**Final choice:**
✔ **Random Forest** for production detection
✔ **Logistic Regression** for explainability & audits

---

## 🎯 Business Value Delivered

* ✔ Reduced false negatives → lower financial loss
* ✔ Controlled false positives → better user experience
* ✔ Scalable pipeline → multi-platform deployment
* ✔ Reproducible experiments → governance & compliance

Absolutely! I can help you craft a **professional, detailed, and cohesive README for Task 3**, integrating Tasks 1 & 2 context and clearly explaining the SHAP analysis you did. Here’s a polished version you can use:

---

# Task 3: Model Interpretation with SHAP

**Developed for Adey Innovations Inc.**

---

## 📌 Overview

Following the successful **data preparation**, **model building**, and **ensemble model evaluation** in Tasks 1 & 2, Task 3 focuses on **interpreting the predictions** of our best-performing model.

Understanding why the model flags certain transactions as fraudulent is critical for:

* **Actionable business insights**
* **Operational transparency**
* **Regulatory compliance in the fintech domain**

I leveraged **SHAP (SHapley Additive exPlanations)** to uncover the key drivers of fraud detection, compare these to built-in feature importance, and generate **data-driven recommendations** to reduce financial loss and improve user experience.

---

## 🧠 Objective

* Interpret the **Random Forest model** predictions to understand the drivers of fraud.
* Identify **top features** influencing fraud detection.
* Provide **actionable business recommendations** based on predictive patterns.
* Ensure the insights align with previously established **modeling rationale** (Tasks 1 & 2).

---

## 🛠️ Methodology

### 1. Feature Importance Baseline

We first examine the **built-in feature importance** from the Random Forest model:

* Visualize the **top 10 features**.
* Identify which features carry the most predictive power.
* Compare baseline importance with SHAP results to validate consistency.

### 2. SHAP Analysis

**SHAP** provides a robust framework to explain **both global and individual predictions**:

* **Global Analysis**: Summary plots reveal the overall importance of features across all transactions.
* **Individual Analysis**: Force plots are generated for:

  * **True Positive:** Correctly identified fraud
  * **False Positive:** Legitimate transaction flagged as fraud
  * **False Negative:** Fraudulent transaction missed by the model

This allows stakeholders to see **why a transaction was flagged**, increasing trust and interpretability.

---

### 3. Interpretation

#### Observation

* `device_id_freq` and `time_since_signup` dominate the model, contributing most predictive power.
* Other features, including IP range, purchase value, time of day, country frequency, and demographics, are minor contributors.

#### Interpretation

* The model primarily leverages **behavioral patterns** rather than static attributes.
* High device reuse signals coordinated fraud.
* Transactions immediately following account creation are highly indicative of fraudulent intent.
* Minor features provide context but do not drive decisions independently.

#### Top 5 Drivers of Fraud Predictions

| Rank | Feature                  | Importance |
| ---- | ------------------------ | ---------- |
| 1    | `device_id_freq`         | ~0.51      |
| 2    | `time_since_signup`      | ~0.37      |
| 3    | `ip_int`                 | ~0.02      |
| 4    | `user_id`                | ~0.02      |
| 5    | `upper_bound_ip_address` | ~0.02      |

**Insights:**

* `device_id_freq` & `ip_int`: Highlight **account farming or bot activity**.
* `time_since_signup`: Detects rapid post-signup fraud tactics (“hit-and-run”).
* Behavioral features outweigh geographic or demographic features due to **adversarial adaptation** by fraudsters (VPNs, proxies, etc.).

#### Counterintuitive Findings

* Initial assumptions expected **geography** to be a key fraud indicator.
* SHAP reveals that **behavioral velocity and account age** are far more predictive.
* Country-based features introduce noise and false positives, while device-level patterns provide precise signals.
* This confirms the model is **intent- and behavior-focused**, improving robustness against VPNs and biased geographic rules.

---

## 📊 Business Recommendations

Based on SHAP insights, we recommend:

1. **Enhanced Verification for New Accounts**

   * Transactions within the first **X hours of signup** should trigger additional verification.
   * SHAP Insight: `time_since_signup` is a top predictor of fraud.

2. **Device Monitoring**

   * Flag accounts with **high device reuse** across multiple transactions.
   * SHAP Insight: `device_id_freq` is the most decisive feature for detecting bot-driven fraud.

3. **IP Risk Management**

   * Implement rules for **high-frequency IP addresses** and investigate unusual access patterns.
   * SHAP Insight: `ip_int` and `upper_bound_ip_address` capture coordinated or automated attacks.

> These recommendations directly translate model understanding into actionable **fraud mitigation strategies**.

---

## ✅ Key Takeaways

* **Model Behavior:** Focused on intent and velocity, not static demographics.
* **Robustness:** Reduces false positives while maintaining high fraud detection.
* **Business Impact:** Insights inform operational rules, improve user experience, and reduce financial leakage.
* **Explainability:** SHAP ensures transparent AI, aligning with compliance and audit requirements.

