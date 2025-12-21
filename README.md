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

---

**Would you like me to generate a summary of the SHAP findings specifically for the Credit Card dataset to help refine your business rules?**