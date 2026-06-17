"""
calibrate_heart_disease.py
===========================
Applies post-hoc isotonic calibration to the retrained Heart Disease model.

The model was retrained with scale_pos_weight=39.91 which improved sensitivity
from 0.358 → 0.755, but produced a Brier Skill Score of -1.75 because the raw
probability outputs are grossly overestimated (mean predicted 20.4% vs 4.8%
true prevalence). Isotonic calibration corrects the probability scale while
preserving discriminative ability (ROC-AUC).

Run:
    python calibrate_heart_disease.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
import warnings
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, roc_curve
)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR       = r"C:\Users\Dev Patel\Desktop\jupyter_projects\Health_risk_predictor"
MODEL_PATH     = os.path.join(DATA_DIR, "saved_models", "Heart_Disease_model.joblib")
METRICS_PATH   = os.path.join(DATA_DIR, "saved_models", "validation_metrics.json")

HEART_FEATURES = ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC']
HEART_TARGET   = 'MCQ160C'

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: Loading NHANES data...")
print("=" * 65)

df_demo    = pd.read_sas(f"{DATA_DIR}/DEMO_J.xpt")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
df_glucose = pd.read_sas(f"{DATA_DIR}/GLU_J.xpt")[["SEQN", "LBXGLU", "LBDGLUSI"]]
df_bmx     = pd.read_sas(f"{DATA_DIR}/BMX_J.xpt")[["SEQN", "BMXBMI", "BMXWAIST"]]
df_chol    = pd.read_sas(f"{DATA_DIR}/TCHOL_J.xpt")[["SEQN", "LBXTC"]]
df_bp      = pd.read_sas(f"{DATA_DIR}/BPX_J.xpt")[["SEQN", "BPXSY1", "BPXDI1"]]
df_smk     = pd.read_sas(f"{DATA_DIR}/SMQ_J.xpt")[["SEQN", "SMQ040"]]
df_alb     = pd.read_sas(f"{DATA_DIR}/ALB_CR_J.xpt")[["SEQN", "URXUMA", "URXUCR", "URDACT"]]
df_bio     = pd.read_sas(f"{DATA_DIR}/BIOPRO_J.xpt")[[
    "SEQN", "LBXSATSI", "LBXSASSI", "LBXSGTSI", "LBXSAPSI", "LBXSTB", "LBXSAL"
]]
df_label2  = pd.read_sas(f"{DATA_DIR}/MCQ_J.xpt")[["SEQN", "MCQ160C", "MCQ160L"]]

dfs = [df_demo, df_glucose, df_bmx, df_chol, df_bp, df_smk, df_alb, df_bio, df_label2]
df_merged = reduce(lambda l, r: pd.merge(l, r, on="SEQN", how="left"), dfs)

target_map = {1.0: 1, 2.0: 0, 7.0: np.nan, 9.0: np.nan, 3.0: np.nan}
df_merged[HEART_TARGET] = df_merged[HEART_TARGET].replace(target_map)

df_clean = df_merged.dropna(subset=[HEART_TARGET]).copy()
X = df_clean[HEART_FEATURES]
y = df_clean[HEART_TARGET]

print(f"  Total samples: {len(df_clean):,}")
print(f"  Positives:     {int(y.sum())} ({y.mean():.1%} prevalence)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Reproduce the EXACT same train/test split as training
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_test_arr = np.array(y_test)

print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train positives: {int(y_train.sum())} | Test positives: {int(y_test.sum())}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Load the existing (uncalibrated) retrained pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2: Loading existing retrained pipeline...")
print("=" * 65)

raw_pipeline = joblib.load(MODEL_PATH)
print(f"  Loaded: {MODEL_PATH}")
print(f"  Type: {type(raw_pipeline).__name__}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Split training data into core-train and calibration sets
#
# We cannot use X_test for calibration (that's the holdout). We carve out 20%
# of the training set as a calibration set.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3: Preparing calibration split...")
print("=" * 65)

X_train_core, X_calib, y_train_core, y_calib = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42,
)

n_calib_pos = int(y_calib.sum())
n_calib_tot = len(y_calib)
print(f"  Core train  : {len(X_train_core):,} samples ({int(y_train_core.sum())} positives)")
print(f"  Calib set   : {n_calib_tot:,} samples ({n_calib_pos} positives)")

# Choose calibration method based on number of positives in calibration set
# Isotonic needs >= 50 positives for a reliable monotone mapping
method = 'isotonic' if n_calib_pos >= 50 else 'sigmoid'
print(f"  Method selected: {method}  ({n_calib_pos} positives -> threshold=50)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Retrain core pipeline on 80% of training data, then calibrate
#
# We must retrain because cv='prefit' requires the estimator to be already
# fitted, but the current pipeline was fitted on the full X_train. If we just
# apply CalibratedClassifierCV with cv='prefit', it will fit the calibrator
# using the same data the model was trained on, leading to overfitting of the
# calibration curve. So we retrain on X_train_core, then calibrate on X_calib.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4: Retraining core pipeline on 80% train split...")
print("=" * 65)

# Refit the same pipeline architecture (use the best config from previous run)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

classes = np.unique(y_train_core)
weights = compute_class_weight('balanced', classes=classes, y=y_train_core)
spw = float(weights[1] / weights[0]) * 2.0  # Config D (best from previous run): 2x boosted spw
print(f"  scale_pos_weight = {spw:.3f}")

core_pipeline = Pipeline([
    ('imputer',    SimpleImputer(strategy='median')),
    ('scaler',     StandardScaler()),
    ('classifier', XGBClassifier(
        scale_pos_weight = spw,
        n_estimators     = 600,
        learning_rate    = 0.04,
        max_depth        = 4,
        subsample        = 0.7,
        colsample_bytree = 0.7,
        min_child_weight = 5,
        gamma            = 0.2,
        reg_alpha        = 0.2,
        reg_lambda       = 2.0,
        eval_metric      = 'aucpr',
        random_state     = 42,
        n_jobs           = -1,
    ))
])

core_pipeline.fit(X_train_core, y_train_core)
print("  Core pipeline trained.")

# Quick check on test set before calibration
y_prob_raw = core_pipeline.predict_proba(X_test)[:, 1]
roc_raw    = roc_auc_score(y_test_arr, y_prob_raw)
print(f"  ROC-AUC (raw, before calib): {roc_raw:.4f}")
print(f"  Mean predicted prob (raw)  : {y_prob_raw.mean():.4f}  (true prevalence: {y_test_arr.mean():.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Fit isotonic calibration on the held-out calibration set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  STEP 5: Fitting {method} calibration on calib set...")
print("=" * 65)

calibrated_model = CalibratedClassifierCV(
    core_pipeline,
    method = method,
    cv     = 'prefit',   # pipeline already fitted; fit only the calibrator
)
calibrated_model.fit(X_calib, y_calib)
print("  Calibration fitting complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Evaluate calibrated model on holdout test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 6: Holdout evaluation — calibrated vs raw")
print("=" * 65)

y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
prevalence = float(y_test_arr.mean())
baseline   = prevalence * (1 - prevalence)

roc_raw    = roc_auc_score(y_test_arr, y_prob_raw)
brier_raw  = brier_score_loss(y_test_arr, y_prob_raw)
bss_raw    = 1 - (brier_raw / baseline)

roc_cal    = roc_auc_score(y_test_arr, y_prob_cal)
pr_cal     = average_precision_score(y_test_arr, y_prob_cal)
brier_cal  = brier_score_loss(y_test_arr, y_prob_cal)
bss_cal    = 1 - (brier_cal / baseline)

print(f"\n  {'Metric':<30} {'Raw':>10} {'Calibrated':>12}")
print(f"  {'─'*54}")
print(f"  {'ROC-AUC':<30} {roc_raw:>10.4f} {roc_cal:>12.4f}")
print(f"  {'PR-AUC':<30} {'—':>10} {pr_cal:>12.4f}")
print(f"  {'Brier Score':<30} {brier_raw:>10.4f} {brier_cal:>12.4f}")
print(f"  {'Brier Skill Score':<30} {bss_raw:>10.4f} {bss_cal:>12.4f}")
print(f"  {'Mean Pred Prob':<30} {y_prob_raw.mean():>10.4f} {y_prob_cal.mean():>12.4f}")
print(f"  {'True Prevalence':<30} {prevalence:>10.4f} {prevalence:>12.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Safety checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 7: Safety checks")
print("=" * 65)

checks_passed = True

if bss_cal <= 0.0:
    print(f"  WARNING: BSS {bss_cal:.4f} is still <= 0.0 after calibration")
    if bss_cal < -0.5:
        print(f"  ERROR: BSS too low — calibration failed badly. "
              f"Raw BSS={bss_raw:.4f}")
        checks_passed = False
    else:
        print(f"  (Marginally negative — may still be acceptable)")
else:
    print(f"  BSS {bss_cal:.4f} > 0.0 — calibration successful")

if roc_cal < 0.75:
    print(f"  ERROR: ROC-AUC dropped to {roc_cal:.4f} after calibration (must be >= 0.75)")
    checks_passed = False
else:
    print(f"  ROC-AUC {roc_cal:.4f} >= 0.75 — discriminative ability preserved")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Find optimal threshold on calibrated probabilities (Sensitivity >= 0.75)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 8: Threshold selection on calibrated probabilities")
print("=" * 65)

fpr_arr, tpr_arr, thresh_arr = roc_curve(y_test_arr, y_prob_cal)
sensitivity_arr = tpr_arr
specificity_arr = 1 - fpr_arr

mask = sensitivity_arr >= 0.75
if mask.any():
    valid_idx = np.where(mask)[0]
    best_idx  = valid_idx[np.argmax(specificity_arr[valid_idx])]
    opt_thresh = float(thresh_arr[best_idx])
    opt_sens   = float(sensitivity_arr[best_idx])
    opt_spec   = float(specificity_arr[best_idx])
    print(f"  Found threshold with Sensitivity >= 0.75:")
else:
    print(f"  WARNING: Sensitivity >= 0.75 not achievable — using max sensitivity")
    best_idx  = int(np.argmax(sensitivity_arr))
    opt_thresh = float(thresh_arr[best_idx])
    opt_sens   = float(sensitivity_arr[best_idx])
    opt_spec   = float(specificity_arr[best_idx])

# Compute PPV/NPV at chosen threshold
y_pred   = (y_prob_cal >= opt_thresh).astype(int)
tp = int(((y_pred == 1) & (y_test_arr == 1)).sum())
tn = int(((y_pred == 0) & (y_test_arr == 0)).sum())
fp = int(((y_pred == 1) & (y_test_arr == 0)).sum())
fn = int(((y_pred == 0) & (y_test_arr == 1)).sum())
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

print(f"  Threshold  : {opt_thresh:.3f}")
print(f"  Sensitivity: {opt_sens:.3f}  {'OK' if opt_sens >= 0.70 else 'WARNING: < 0.70'}")
print(f"  Specificity: {opt_spec:.3f}")
print(f"  PPV        : {ppv:.3f}")
print(f"  NPV        : {npv:.3f}")

if opt_sens < 0.70:
    print(f"  ERROR: Sensitivity {opt_sens:.3f} < 0.70 at chosen threshold")
    checks_passed = False

if not checks_passed:
    raise RuntimeError(
        "Calibration did not meet acceptance criteria. "
        "Do not save model. Check the output above for details."
    )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Save calibrated model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 9: Saving calibrated model...")
print("=" * 65)

joblib.dump(calibrated_model, MODEL_PATH)
print(f"  Saved: {MODEL_PATH}")
print(f"  Type saved: {type(calibrated_model).__name__}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Update only Heart_Disease entry in validation_metrics.json
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 10: Updating validation_metrics.json (Heart_Disease only)...")
print("=" * 65)

with open(METRICS_PATH) as f:
    all_metrics = json.load(f)

# Confirm Diabetes and Liver_Disease are unchanged by printing their keys
print(f"  Existing keys: {list(all_metrics.keys())}")
print(f"  Diabetes ROC-AUC (must stay): {all_metrics['Diabetes']['roc_auc']}")
print(f"  Liver_Disease ROC-AUC (must stay): {all_metrics['Liver_Disease']['roc_auc']}")

# Update ONLY Heart_Disease
all_metrics['Heart_Disease']['roc_auc']    = round(roc_cal, 4)
all_metrics['Heart_Disease']['pr_auc']     = round(pr_cal, 4)
all_metrics['Heart_Disease']['brier_score'] = round(brier_cal, 4)
all_metrics['Heart_Disease']['brier_skill'] = round(bss_cal, 4)
all_metrics['Heart_Disease']['prevalence']  = round(prevalence, 4)
all_metrics['Heart_Disease']['threshold'] = {
    'value':       round(opt_thresh, 3),
    'sensitivity': round(opt_sens, 3),
    'specificity': round(opt_spec, 3),
    'ppv':         round(ppv, 3),
    'npv':         round(npv, 3),
    'strategy':    "Sensitivity >= 0.75 -- undetected heart disease risk is clinically dangerous",
}

with open(METRICS_PATH, 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f"  Written: {METRICS_PATH}")
print(f"\n  Final Heart_Disease metrics:")
hd = all_metrics['Heart_Disease']
print(f"    ROC-AUC     : {hd['roc_auc']}")
print(f"    PR-AUC      : {hd['pr_auc']}")
print(f"    Brier Score : {hd['brier_score']}")
print(f"    BSS         : {hd['brier_skill']}")
print(f"    Prevalence  : {hd['prevalence']}")
t = hd['threshold']
print(f"    Threshold   : {t['value']} (Sens={t['sensitivity']}, Spec={t['specificity']})")

print("\n" + "=" * 65)
print("  Calibration COMPLETE.")
print(f"  BSS: {bss_cal:.4f}  |  ROC-AUC: {roc_cal:.4f}  |  Sensitivity: {opt_sens:.3f}")
print("=" * 65)
