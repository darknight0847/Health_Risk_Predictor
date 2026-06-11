"""
test_holdout_evaluation.py
==========================
Formal holdout evaluation of the 3 trained health-risk models using
the original NHANES .xpt data files.

IMPORTANT: This reproduces the EXACT same train/test split used during
training (random_state=42, test_size=0.2, stratify=y), so the test set
is genuinely unseen by the model during its final fit.

Metrics computed per model:
  - ROC-AUC
  - Average Precision (PR-AUC)
  - Accuracy, Precision, Recall, F1 (at 0.5 threshold)
  - Confusion Matrix
  - Calibration: Mean predicted prob vs actual prevalence

Run:
    python test_holdout_evaluation.py
"""

import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\Dev Patel\Desktop\jupyter_projects\Health_risk_predictor"

MODELS_CONFIG = {
    "Diabetes": {
        "model_path": "saved_models/Diabetes_model.joblib",
        "target":     "DIQ010",
        "features":   ["RIDAGEYR", "LBDGLUSI", "LBXGLU", "BMXWAIST", "BMXBMI", "BPXSY1", "URDACT"],
    },
    "Heart Disease": {
        "model_path": "saved_models/Heart_Disease_model.joblib",
        "target":     "MCQ160C",
        "features":   ["RIDAGEYR", "URDACT", "BMXWAIST", "RIAGENDR", "LBXTC"],
    },
    "Liver Disease": {
        "model_path": "saved_models/Liver_Disease_model.joblib",
        "target":     "MCQ160L",
        # Updated features: original 5 + 6 liver enzyme markers from BIOPRO_J.xpt
        "features":   [
            "RIDAGEYR", "LBXGLU", "LBDGLUSI", "BPXSY1", "URXUMA",
            "LBXSATSI",   # ALT  (Alanine Aminotransferase)
            "LBXSASSI",   # AST  (Aspartate Aminotransferase)
            "LBXSGTSI",   # GGT  (Gamma-glutamyl transferase)
            "LBXSAPSI",   # ALP  (Alkaline Phosphatase)
            "LBXSTB",     # Total Bilirubin
            "LBXSAL",     # Serum Albumin
        ],
    },
}

# ─────────────────────────────────────────────
# STEP 1: Load & merge data (same as training)
# ─────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: Loading and merging NHANES data...")
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
]]  # Liver enzyme markers
df_label1  = pd.read_sas(f"{DATA_DIR}/DIQ_J.xpt")[["SEQN", "DIQ010"]]
df_label2  = pd.read_sas(f"{DATA_DIR}/MCQ_J.xpt")[["SEQN", "MCQ160C", "MCQ160L"]]

dfs = [df_demo, df_glucose, df_bmx, df_chol, df_bp, df_smk, df_alb, df_bio, df_label1, df_label2]
df_merged = reduce(lambda l, r: pd.merge(l, r, on="SEQN", how="left"), dfs)

# Clean targets - same mapping as training
target_map = {1.0: 1, 2.0: 0, 7.0: np.nan, 9.0: np.nan, 3.0: np.nan}
for t in ["DIQ010", "MCQ160C", "MCQ160L"]:
    df_merged[t] = df_merged[t].replace(target_map)

print(f"Merged dataset: {len(df_merged):,} total records\n")

# ─────────────────────────────────────────────
# STEP 2: Evaluate each model
# ─────────────────────────────────────────────
print("=" * 65)
print("  STEP 2: Evaluating models on held-out test sets")
print("=" * 65)

all_results = {}

for disease, cfg in MODELS_CONFIG.items():
    print(f"\n{'─'*65}")
    print(f"  {disease}")
    print(f"{'─'*65}")

    # Reproduce EXACT train/test split from training
    df_clean = df_merged.dropna(subset=[cfg["target"]]).copy()
    X = df_clean[cfg["features"]]
    y = df_clean[cfg["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")
    print(f"  Positives in test: {int(y_test.sum())} ({y_test.mean():.1%} prevalence)")

    # Load model and predict on test set
    model  = joblib.load(cfg["model_path"])
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Core metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    # Calibration
    mean_pred   = y_prob.mean()
    actual_prev = y_test.mean()

    # Best threshold by F1
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f  = f1_score(y_test, yp, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    best_pred = (y_prob >= best_thresh).astype(int)
    best_prec = precision_score(y_test, best_pred, zero_division=0)
    best_rec  = recall_score(y_test, best_pred, zero_division=0)

    all_results[disease] = {
        "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
        "Accuracy": acc, "Precision@0.5": prec, "Recall@0.5": rec, "F1@0.5": f1,
        "Best_Thresh": best_thresh, "Best_F1": best_f1,
        "Best_Prec": best_prec, "Best_Rec": best_rec,
        "Mean_Pred_Prob": mean_pred, "Actual_Prevalence": actual_prev,
        "Test_Size": len(X_test), "Positives": int(y_test.sum()),
        "Confusion_Matrix": cm,
    }

    # Print metrics
    print(f"\n  {'Metric':<28} {'Value':>10}")
    print(f"  {'─'*40}")
    print(f"  {'ROC-AUC':<28} {roc_auc:>10.4f}")
    print(f"  {'PR-AUC (Avg Precision)':<28} {pr_auc:>10.4f}")
    print(f"  {'Accuracy (@ 0.5)':<28} {acc:>10.1%}")
    print(f"  {'Precision (@ 0.5)':<28} {prec:>10.1%}")
    print(f"  {'Recall    (@ 0.5)':<28} {rec:>10.1%}")
    print(f"  {'F1 Score  (@ 0.5)':<28} {f1:>10.4f}")
    print(f"  {'─'*40}")
    print(f"  {'Best Threshold (by F1)':<28} {best_thresh:>10.2f}")
    print(f"  {'Best F1':<28} {best_f1:>10.4f}")
    print(f"  {'Best Precision':<28} {best_prec:>10.1%}")
    print(f"  {'Best Recall':<28} {best_rec:>10.1%}")
    print(f"  {'─'*40}")
    print(f"  {'Mean Predicted Prob':<28} {mean_pred:>10.1%}")
    print(f"  {'Actual Prevalence':<28} {actual_prev:>10.1%}")
    cal_diff = abs(mean_pred - actual_prev)
    if   cal_diff < 0.05: cal_note = "Well-calibrated"
    elif cal_diff < 0.10: cal_note = "Slightly off"
    else:                 cal_note = "Poorly calibrated"
    print(f"  {'Calibration':<28} {cal_diff:>8.1%}  ({cal_note})")

    # Confusion matrix
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    print(f"\n  Confusion Matrix (@ 0.5 threshold):")
    print(f"  +-----------------------------+")
    print(f"  |              Pred No  Pred Yes|")
    print(f"  | Actual No :  {tn:>7}  {fp:>7} |")
    print(f"  | Actual Yes:  {fn:>7}  {tp:>7} |")
    print(f"  +-----------------------------+")

# ─────────────────────────────────────────────
# STEP 3: Summary Table
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  STEP 3: SUMMARY TABLE")
print(f"{'=' * 65}")
print(f"\n  {'Disease':<20} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1@0.5':>8} {'BestF1':>8} {'CalibErr':>9}")
print(f"  {'─'*65}")
for disease, r in all_results.items():
    cal_diff = abs(r["Mean_Pred_Prob"] - r["Actual_Prevalence"])
    print(f"  {disease:<20} {r['ROC-AUC']:>8.4f} {r['PR-AUC']:>8.4f} "
          f"{r['F1@0.5']:>8.4f} {r['Best_F1']:>8.4f} {cal_diff:>8.1%}")

print(f"\n  Interpretation Guide:")
print(f"  ROC-AUC  >0.80 = Good  |  >0.90 = Excellent")
print(f"  PR-AUC   higher is better (especially with class imbalance)")
print(f"  CalibErr <5%   = Well-calibrated (probabilities are reliable)")
print(f"\n{'=' * 65}")
print("  Holdout evaluation complete.")
print(f"{'=' * 65}\n")
