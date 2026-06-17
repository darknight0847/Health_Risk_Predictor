---
title: Health Risk Predictor
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# MedScan AI — Multi-Disease Health Risk Predictor


MedScan AI is an educational clinical decision support tool that predicts the risk probability of three major diseases—**Diabetes**, **Heart Disease**, and **Liver Disease**—using custom-calibrated XGBoost models. Every prediction is fully explained using local **SHAP (SHAPley Additive exPlanations)** waterfall analysis to highlight which patient vitals are driving the risk assessment.

---

## 🚀 Key Features

*   **Multi-Disease Risk Engine**: Predicts risks simultaneously using clinical biomarkers.
*   **SHAP Waterfall Analysis**: Dynamically generates local feature contribution plots showing how much each input pushes the risk level up (red) or down (blue) relative to the population baseline.
*   **Premium Glassmorphic UI**: Real-time CSS validation of normal, borderline, and pathological clinical ranges for all 15 inputs.
*   **Optimal Performance**: Leverages sampled reference backgrounds to generate full SHAP assessments in under a second.

---

## 📈 Model Performance & Validation
All models are trained on the CDC's **NHANES** (National Health and Nutrition Examination Survey) dataset and validated against a stratified holdout split of **9,254 records**.

→ [Model Card](./MODEL_CARD.md)

| Disease | ROC-AUC | PR-AUC | Brier Score | Threshold | Sensitivity | Specificity | Threshold Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Diabetes** | **0.91** | **0.6171** | **0.0578** | **0.098** | **0.860** | **0.791** | Sensitivity ≥ 0.85 |
| **Heart Disease** | **0.80** | **0.1888** | **0.0429** | **0.026** | **0.755** | **0.710** | Sensitivity ≥ 0.75 |
| **Liver Disease** | **0.75** | **0.1712** | **0.0476** | **0.070** | **0.542** | **0.802** | Specificity ≥ 0.80 |

> **PR-AUC** is more honest than ROC-AUC for rare-disease prediction (class imbalance). **Brier Score** measures calibration quality (lower is better). Heart Disease uses post-hoc Platt calibration to align risk outputs. Brier Skill Score normalises calibration against base rate (Diabetes: 0.3727, Heart Disease: 0.0567, Liver Disease: 0.0538).

---

## 📂 Repository Structure

```text
├── flask_app/
│   ├── app.py                           # Flask Backend (Loads models & computes SHAP plots)
│   ├── requirements.txt                 # Backend dependencies (pinned versions)
│   ├── generate_calibration_plots.py    # One-time script: generates calibration curve PNGs
│   ├── static/calibration/              # Pre-generated calibration PNG files
│   └── templates/
│       └── index.html                   # Premium frontend template with range validation & autofill
├── saved_models/
│   ├── Diabetes_model.joblib            # Trained pipeline (imputer + scaler + XGBoost)
│   ├── Heart_Disease_model.joblib
│   ├── Liver_Disease_model.joblib
│   ├── background_data.joblib           # Sampled background distributions for SHAP explainer
│   └── validation_metrics.json         # Pre-computed holdout metrics (ROC-AUC, PR-AUC, Brier, thresholds)
├── test_clinical_personas.py            # Smoke tests for 10 hand-crafted clinical personas
├── test_holdout_evaluation.py           # Validation script computing formal test split metrics
└── .gitignore                           # Keeps large raw NHANES SAS files (*.xpt) off GitHub
```

---

## 🛠️ Local Quickstart

### 1. Clone & Set Up Directory
Ensure the directory structure matches the repository layout.

### 2. Install Dependencies
```bash
pip install -r flask_app/requirements.txt
```

### 3. Access the App

Visit the live application here:

https://darknight0847-health-risk-predictor.hf.space/

Or click:

[Health Risk Predictor](https://darknight0847-health-risk-predictor.hf.space/)

> [!TIP]
> Use the **🟢 Autofill (Healthy)** and **🔴 Autofill (High Risk)** buttons in the web interface to instantly test different diagnostic scenarios!

---

## ⚖️ Educational Disclaimer
MedScan AI is a machine learning research prototype. It is designed solely for educational purposes and **does not constitute professional medical advice, diagnosis, or treatment**.
