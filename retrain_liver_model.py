"""
retrain_liver_model.py
======================
Retrains ONLY the Liver Disease model with an expanded feature set
that includes liver enzyme markers from BIOPRO_J.xpt.

New features added (all from BIOPRO_J.xpt):
  - LBXSATSI : ALT  (Alanine Aminotransferase)  -- strongest liver signal
  - LBXSASSI : AST  (Aspartate Aminotransferase) -- strong liver signal
  - LBXSGTSI : GGT  (Gamma-glutamyl transferase)  -- strong liver signal
  - LBXSAPSI : ALP  (Alkaline Phosphatase)
  - LBXSTB   : Total Bilirubin
  - LBXSAL   : Serum Albumin  (low = liver failure marker)

Kept from original model:
  - RIDAGEYR, LBXGLU, BPXSY1, URXUMA, LBDGLUSI

Run:
    python -X utf8 retrain_liver_model.py
"""

import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
import optuna

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR       = r"C:\Users\Dev Patel\Desktop\jupyter_projects\Health_risk_predictor"
N_TRIALS       = 80   # more than original 30 for better search
TARGET_COL     = "MCQ160L"
SAVE_PATH      = "saved_models/Liver_Disease_model.joblib"
SAVE_PATH_BKP  = "saved_models/Liver_Disease_model_OLD.joblib"

# All features for new Liver Disease model
LIVER_FEATURES = [
    # Original features
    "RIDAGEYR",   # Age
    "LBXGLU",     # Blood glucose (mg/dL)
    "LBDGLUSI",   # Blood glucose (mmol/L)
    "BPXSY1",     # Systolic BP
    "URXUMA",     # Urine albumin
    # NEW: Liver enzyme markers from BIOPRO_J.xpt
    "LBXSATSI",   # ALT  -- key liver enzyme
    "LBXSASSI",   # AST  -- key liver enzyme
    "LBXSGTSI",   # GGT  -- liver/bile marker
    "LBXSAPSI",   # ALP  -- alkaline phosphatase
    "LBXSTB",     # Total bilirubin -- jaundice marker
    "LBXSAL",     # Serum albumin -- low = liver failure
]

# ─────────────────────────────────────────────
# STEP 1: Load & merge data
# ─────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: Loading and merging NHANES data...")
print("=" * 65)

df_demo    = pd.read_sas(f"{DATA_DIR}/DEMO_J.xpt")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
df_glucose = pd.read_sas(f"{DATA_DIR}/GLU_J.xpt")[["SEQN", "LBXGLU", "LBDGLUSI"]]
df_bp      = pd.read_sas(f"{DATA_DIR}/BPX_J.xpt")[["SEQN", "BPXSY1", "BPXDI1"]]
df_alb     = pd.read_sas(f"{DATA_DIR}/ALB_CR_J.xpt")[["SEQN", "URXUMA", "URDACT"]]
df_bio     = pd.read_sas(f"{DATA_DIR}/BIOPRO_J.xpt")[[
    "SEQN", "LBXSATSI", "LBXSASSI", "LBXSGTSI",
    "LBXSAPSI", "LBXSTB", "LBXSAL"
]]
df_label   = pd.read_sas(f"{DATA_DIR}/MCQ_J.xpt")[["SEQN", "MCQ160L"]]

dfs = [df_demo, df_glucose, df_bp, df_alb, df_bio, df_label]
df_merged = reduce(lambda l, r: pd.merge(l, r, on="SEQN", how="left"), dfs)

target_map = {1.0: 1, 2.0: 0, 7.0: np.nan, 9.0: np.nan, 3.0: np.nan}
df_merged[TARGET_COL] = df_merged[TARGET_COL].replace(target_map)

df_clean = df_merged.dropna(subset=[TARGET_COL]).copy()
X = df_clean[LIVER_FEATURES]
y = df_clean[TARGET_COL]

print(f"  Dataset: {len(df_clean):,} records   |   Positives: {int(y.sum())} ({y.mean():.1%})")
print(f"  Features ({len(LIVER_FEATURES)}): {LIVER_FEATURES}")

# ─────────────────────────────────────────────
# STEP 2: Train / Test split (same params as original)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train):,}   Test: {len(X_test):,}")

# Class imbalance weight
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
dynamic_weight = np.sqrt(num_neg / num_pos)
print(f"  Class imbalance weight: {dynamic_weight:.2f}")

# Background data for SHAP (100 samples from train)
import shap
background_data_liver = shap.sample(X_train, 100)

# ─────────────────────────────────────────────
# STEP 3: Optuna hyperparameter search
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  STEP 2: Optuna hyperparameter search ({N_TRIALS} trials)...")
print(f"{'=' * 65}")

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 9),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight":  dynamic_weight,
        "eval_metric":       "logloss",
        "random_state":      42,
        "n_jobs":            -1,
    }
    pipeline = Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("xgb",        XGBClassifier(**params)),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params
best_params["scale_pos_weight"] = dynamic_weight
best_params["eval_metric"]      = "logloss"
best_params["random_state"]     = 42
best_params["n_jobs"]           = -1

print(f"  Best ROC-AUC (search): {study.best_value:.4f}")
print(f"  Best params: {best_params}")

# ─────────────────────────────────────────────
# STEP 4: Train final calibrated model
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  STEP 3: Training final calibrated model...")
print(f"{'=' * 65}")

base_xgb      = XGBClassifier(**best_params)
calibrated    = CalibratedClassifierCV(base_xgb, method="isotonic", cv=5)
final_pipeline = Pipeline([
    ("imputer",    SimpleImputer(strategy="median")),
    ("scaler",     StandardScaler()),
    ("classifier", calibrated),
])
final_pipeline.fit(X_train, y_train)

y_prob = final_pipeline.predict_proba(X_test)[:, 1]
roc    = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

# Best threshold search
best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.05, 0.80, 0.05):
    yp  = (y_prob >= t).astype(int)
    f   = f1_score(y_test, yp, zero_division=0)
    if f > best_f1:
        best_f1, best_t = f, t

best_pred = (y_prob >= best_t).astype(int)

print(f"\n  +-------------------------------------------------+")
print(f"  |           FINAL MODEL PERFORMANCE               |")
print(f"  +-------------------------------------------------+")
print(f"  | {'ROC-AUC':<28} {roc:>10.4f}         |")
print(f"  | {'PR-AUC (Avg Precision)':<28} {pr_auc:>10.4f}         |")
print(f"  | {'Best F1 Threshold':<28} {best_t:>10.2f}         |")
print(f"  | {'Best F1':<28} {best_f1:>10.4f}         |")
print(f"  | {'Precision @ best thresh':<28} {precision_score(y_test, best_pred, zero_division=0):>10.1%}         |")
print(f"  | {'Recall    @ best thresh':<28} {recall_score(y_test, best_pred, zero_division=0):>10.1%}         |")
calib = abs(y_prob.mean() - y_test.mean())
print(f"  | {'Calibration Error':<28} {calib:>10.1%}         |")
print(f"  +-------------------------------------------------+")

# ─────────────────────────────────────────────
# STEP 5: Backup old model, save new one
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  STEP 4: Saving model...")
print(f"{'=' * 65}")

# Back up old model first
import shutil
if os.path.exists(SAVE_PATH):
    shutil.copy(SAVE_PATH, SAVE_PATH_BKP)
    print(f"  Old model backed up -> {SAVE_PATH_BKP}")

joblib.dump(final_pipeline, SAVE_PATH)
print(f"  New model saved      -> {SAVE_PATH}")

# Update background_data.joblib with new liver background
all_bg = joblib.load("saved_models/background_data.joblib")
all_bg["Liver Disease"] = background_data_liver
joblib.dump(all_bg, "saved_models/background_data.joblib")
print(f"  Background data updated in saved_models/background_data.joblib")

print(f"\n{'=' * 65}")
print("  DONE. Run test_holdout_evaluation.py to verify improvement.")
print(f"{'=' * 65}\n")
