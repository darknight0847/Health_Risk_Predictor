# MedScan AI — Multi-Disease Health Risk Predictor


MedScan AI is an educational clinical decision support tool that predicts the risk probability of three major diseases—**Diabetes**, **Heart Disease**, and **Liver Disease**—using custom-calibrated XGBoost models. Every prediction is fully explained using local **SHAP (SHAPley Additive exPlanations)** waterfall analysis to highlight which patient vitals are driving the risk assessment.

---

## 🚀 Key Features

*   **Multi-Disease Risk Engine**: Predicts risks simultaneously using clinical biomarkers.
*   **SHAP Waterfall Analysis**: Dynamically generates local feature contribution plots showing how much each input pushes the risk level up (red) or down (blue) relative to the population baseline.
*   **Premium Glassmorphic UI**: Real-time CSS validation of normal, borderline, and pathological clinical ranges for all 15 inputs.
*   **Optimal Performance**: Leverages sampled reference backgrounds to generate full SHAP assessments in under a second.

---

## 📈 Model Performance & Features
All models are trained on the CDC's **NHANES** (National Health and Nutrition Examination Survey) dataset and validated against a holdout test split of **9,254 records**.

| Disease | ROC-AUC | Key Clinical Features Used |
| :--- | :---: | :--- |
| **Diabetes** | **0.91** | Age, Fasting Glucose (mg/dL & mmol/L), Waist Circumference, BMI, Systolic Blood Pressure, Albumin-Creatinine Ratio (ACR) |
| **Heart Disease** | **0.84** | Age, Urinary ACR, Waist Circumference, Biological Sex, Total Cholesterol |
| **Liver Disease** | **0.75** | Age, Fasting Glucose, Systolic BP, Urine Albumin, ALT, AST, GGT, ALP, Total Bilirubin, Serum Albumin |

---

## 📂 Repository Structure

```text
├── flask_app/
│   ├── app.py                  # Flask Backend (Loads models & computes SHAP plots)
│   ├── requirements.txt        # Backend dependencies
│   └── templates/
│       └── index.html          # Premium frontend template with range validation & autofill
├── saved_models/
│   ├── Diabetes_model.joblib   # Trained pipeline (imputer + scaler + XGBoost)
│   ├── Heart_Disease_model.joblib
│   ├── Liver_Disease_model.joblib
│   └── background_data.joblib  # Sampled background distributions for SHAP explainer
├── test_clinical_personas.py   # Smoke tests for 10 hand-crafted clinical personas
├── test_holdout_evaluation.py  # Validation script computing formal test split metrics
└── .gitignore                  # Keeps large raw NHANES SAS files (*.xpt) off GitHub
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
