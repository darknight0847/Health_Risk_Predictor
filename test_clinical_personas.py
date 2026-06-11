"""
test_clinical_personas.py
=========================
Sanity / smoke test for the 3 trained health-risk models using
hand-crafted clinical personas. No real patient data needed.

Each persona is designed so we KNOW what the model should output
(low or high risk), letting us verify model direction and calibration.

Run:
    python test_clinical_personas.py
"""

import joblib
import pandas as pd
import numpy as np
import sys

# Force UTF-8 output so emojis don't crash on Windows cp1252 terminals
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────
print("Loading models...")
diabetes_model = joblib.load("saved_models/Diabetes_model.joblib")
heart_model    = joblib.load("saved_models/Heart_Disease_model.joblib")
liver_model    = joblib.load("saved_models/Liver_Disease_model.joblib")
print("✅ Models loaded.\n")

# ─────────────────────────────────────────────
# Helper: predict for all 3 models
# ─────────────────────────────────────────────
def predict_all(patient: dict) -> dict:
    """Given a dict of patient vitals, return risk probabilities."""

    glucose_mgdl  = patient.get("LBXGLU",   np.nan)
    glucose_mmoll = patient.get("LBDGLUSI",
                        glucose_mgdl * 0.0555 if not np.isnan(glucose_mgdl) else np.nan)

    full = {
        "RIDAGEYR":  patient.get("RIDAGEYR",  np.nan),
        "LBXGLU":   glucose_mgdl,
        "LBDGLUSI": glucose_mmoll,
        "BMXWAIST": patient.get("BMXWAIST",  np.nan),
        "BMXBMI":   patient.get("BMXBMI",    np.nan),
        "BPXSY1":   patient.get("BPXSY1",    np.nan),
        "URDACT":   patient.get("URDACT",    np.nan),
        "RIAGENDR": patient.get("RIAGENDR",  np.nan),
        "LBXTC":    patient.get("LBXTC",     np.nan),
        "URXUMA":   patient.get("URXUMA",    np.nan),
    }

    d_feat = ["RIDAGEYR", "LBDGLUSI", "LBXGLU", "BMXWAIST", "BMXBMI", "BPXSY1", "URDACT"]
    h_feat = ["RIDAGEYR", "URDACT", "BMXWAIST", "RIAGENDR", "LBXTC"]
    l_feat = ["RIDAGEYR", "LBXGLU", "BPXSY1", "URXUMA", "LBDGLUSI"]

    df_d = pd.DataFrame([{k: full[k] for k in d_feat}])
    df_h = pd.DataFrame([{k: full[k] for k in h_feat}])
    df_l = pd.DataFrame([{k: full[k] for k in l_feat}])

    return {
        "Diabetes":      diabetes_model.predict_proba(df_d)[0][1],
        "Heart Disease": heart_model.predict_proba(df_h)[0][1],
        "Liver Disease": liver_model.predict_proba(df_l)[0][1],
    }

# ─────────────────────────────────────────────
# 10 Clinical Personas
# ─────────────────────────────────────────────
personas = [
    {
        "name":        "P01 - Young Healthy Male",
        "description": "25yo athletic male, all vitals normal",
        "expected":    "ALL LOW",
        "data": {"RIDAGEYR":25, "RIAGENDR":1, "LBXGLU":90,  "BMXWAIST":80,  "BMXBMI":22.0, "BPXSY1":112, "URDACT":5.0,  "LBXTC":160, "URXUMA":5.0},
    },
    {
        "name":        "P02 - Healthy Middle-aged Female",
        "description": "47yo female, normal BMI, normal labs",
        "expected":    "ALL LOW",
        "data": {"RIDAGEYR":47, "RIAGENDR":2, "LBXGLU":93,  "BMXWAIST":82,  "BMXBMI":23.5, "BPXSY1":116, "URDACT":6.0,  "LBXTC":175, "URXUMA":7.0},
    },
    {
        "name":        "P03 - Classic Type-2 Diabetic",
        "description": "58yo obese male, fasting glucose 240, high waist, high ACR",
        "expected":    "DIABETES HIGH",
        "data": {"RIDAGEYR":58, "RIAGENDR":1, "LBXGLU":240, "BMXWAIST":125, "BMXBMI":37.0, "BPXSY1":138, "URDACT":90.0, "LBXTC":210, "URXUMA":120.0},
    },
    {
        "name":        "P04 - Pre-diabetic Overweight Female",
        "description": "52yo female, borderline glucose 118, overweight",
        "expected":    "DIABETES MODERATE",
        "data": {"RIDAGEYR":52, "RIAGENDR":2, "LBXGLU":118, "BMXWAIST":100, "BMXBMI":30.5, "BPXSY1":125, "URDACT":12.0, "LBXTC":195, "URXUMA":15.0},
    },
    {
        "name":        "P05 - High Cardiac Risk Elder Male",
        "description": "70yo male, high cholesterol 295, high ACR, large waist",
        "expected":    "HEART DISEASE HIGH",
        "data": {"RIDAGEYR":70, "RIAGENDR":1, "LBXGLU":105, "BMXWAIST":128, "BMXBMI":31.0, "BPXSY1":158, "URDACT":150.0,"LBXTC":295, "URXUMA":200.0},
    },
    {
        "name":        "P06 - Hypertensive but Normal Labs",
        "description": "62yo male, high BP 165 but normal glucose/cholesterol",
        "expected":    "MIXED - moderate heart, low diabetes",
        "data": {"RIDAGEYR":62, "RIAGENDR":1, "LBXGLU":98,  "BMXWAIST":102, "BMXBMI":28.0, "BPXSY1":165, "URDACT":10.0, "LBXTC":172, "URXUMA":9.0},
    },
    {
        "name":        "P07 - Liver Disease Risk Profile",
        "description": "49yo male, elevated urine albumin, high glucose, high BP",
        "expected":    "LIVER DISEASE HIGH",
        "data": {"RIDAGEYR":49, "RIAGENDR":1, "LBXGLU":195, "BMXWAIST":110, "BMXBMI":33.0, "BPXSY1":152, "URDACT":55.0, "LBXTC":220, "URXUMA":250.0},
    },
    {
        "name":        "P08 - Elderly with Multiple Risk Factors",
        "description": "75yo female, obese, diabetic range glucose, high chol, high ACR",
        "expected":    "ALL HIGH",
        "data": {"RIDAGEYR":75, "RIAGENDR":2, "LBXGLU":280, "BMXWAIST":135, "BMXBMI":40.0, "BPXSY1":162, "URDACT":200.0,"LBXTC":300, "URXUMA":300.0},
    },
    {
        "name":        "P09 - Child (Edge Case: Age 12)",
        "description": "12yo child, normal vitals - model should extrapolate carefully",
        "expected":    "ALL VERY LOW (edge case)",
        "data": {"RIDAGEYR":12, "RIAGENDR":1, "LBXGLU":88,  "BMXWAIST":64,  "BMXBMI":17.0, "BPXSY1":100, "URDACT":4.0,  "LBXTC":140, "URXUMA":4.0},
    },
    {
        "name":        "P10 - Missing Data Patient",
        "description": "45yo, only age and gender known - model should impute and still predict",
        "expected":    "NEAR MEDIAN RISK (imputed)",
        "data": {"RIDAGEYR":45, "RIAGENDR":1},
    },
]

# ─────────────────────────────────────────────
# Run tests & display results
# ─────────────────────────────────────────────
def risk_label(p):
    if p > 0.5:   return "[HIGH    ]"
    if p > 0.2:   return "[MODERATE]"
    return             "[LOW     ]"

print("=" * 75)
print("        CLINICAL PERSONA TEST RESULTS - HEALTH RISK PREDICTOR")
print("=" * 75)

for persona in personas:
    results = predict_all(persona["data"])

    print(f"\n{'─'*75}")
    print(f"  {persona['name']}")
    print(f"  {persona['description']}")
    print(f"  Expected: {persona['expected']}")
    print(f"{'─'*75}")

    for disease, prob in results.items():
        label = risk_label(prob)
        filled = int(prob * 28)
        bar = "#" * filled + "-" * (28 - filled)
        print(f"  {disease:<16} | {label} | {prob:5.1%}  [{bar}]")

print(f"\n{'=' * 75}")
print(f"  All {len(personas)} personas processed successfully.")
print(f"  Review risk directions above against 'Expected' column.")
print("=" * 75)
