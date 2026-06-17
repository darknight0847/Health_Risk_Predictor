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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Extended Metrics — Brier Score, Brier Skill Score, Optimal Thresholds
# ─────────────────────────────────────────────────────────────────────────────
# WHY THESE METRICS:
#   - Brier Skill Score (BSS): The CORRECT way to interpret Brier Score for
#     imbalanced clinical data. Formula: 1 - (brier / baseline_brier), where
#     baseline_brier = prevalence * (1 - prevalence) (score of a naive model).
#     Interpretation: 0 = no better than naive, 1 = perfect, <0 = worse than naive.
#     Raw Brier Score alone is MISLEADING for rare diseases — a model that always
#     predicts near-zero achieves low Brier without any discriminative power.
#   - PR-AUC (average_precision_score): More honest than ROC-AUC when classes
#     are imbalanced. Measures precision vs recall tradeoff for rare-disease settings.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

print(f"{'=' * 65}")
print("  STEP 4: Extended Metrics + Optimal Thresholds")
print(f"{'=' * 65}")

# Maps the human-readable disease names used in MODELS_CONFIG → JSON keys
DISEASE_KEY_MAP = {
    "Diabetes":      "Diabetes",
    "Heart Disease": "Heart_Disease",
    "Liver Disease": "Liver_Disease",
}


def compute_extended_metrics(model, X_test, y_test, disease_name):
    """
    Compute and return all validation metrics for a single disease model.

    Returns a dict with keys: roc_auc, pr_auc, brier_score, brier_skill, prevalence

    NOTE on Brier Skill Score:
    Raw Brier Score alone is misleading for rare diseases. A model predicting
    the prevalence rate for every patient achieves brier = p*(1-p) without any
    discriminative power. The Brier Skill Score normalises against this naive
    baseline so that 0 = no better than naive and 1 = perfect calibration.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_arr = np.array(y_test)

    roc_auc = roc_auc_score(y_test_arr, y_prob)
    pr_auc  = average_precision_score(y_test_arr, y_prob)
    brier   = brier_score_loss(y_test_arr, y_prob)

    # ── Brier Skill Score — correct interpretation for imbalanced clinical data ──
    # Formula: 1 - (model_brier / baseline_brier)
    # baseline_brier = prevalence * (1 - prevalence)  [naive model that always
    #                                                   predicts the base rate]
    # Interpretation:
    #   1.0  = perfect
    #   0.0  = same as naive model (always predicts prevalence)
    #  <0.0  = worse than naive — serious problem
    prevalence     = float(y_test_arr.mean())
    brier_baseline = prevalence * (1 - prevalence)
    brier_skill    = 1.0 - (brier / brier_baseline) if brier_baseline > 0 else 0.0

    print(f"\n{'=' * 40}")
    print(f"  {disease_name}")
    print(f"{'=' * 40}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"  PR-AUC            : {pr_auc:.4f}")
    print(f"  Disease prevalence : {prevalence:.3f}")
    print(f"  Brier Score        : {brier:.4f}  (raw — do not interpret in isolation)")
    print(f"  Brier baseline     : {brier_baseline:.4f}  (naive model score)")
    print(f"  Brier Skill Score  : {brier_skill:.4f}  (0=useless, 1=perfect)")

    if brier_skill < 0.10:
        print(f"  ⚠️  WARNING: Brier Skill Score < 0.10 — model barely beats naive baseline")
    elif brier_skill < 0.25:
        print(f"  ⚠️  WEAK: Model has limited probabilistic skill")
    else:
        print(f"  ✅ Acceptable probabilistic skill")

    return {
        "roc_auc":    roc_auc,
        "pr_auc":     pr_auc,
        "brier_score": brier,
        "brier_skill": round(brier_skill, 4),
        "prevalence":  round(prevalence, 4),
    }


def find_optimal_threshold(y_test, y_prob, disease_name):
    """
    Threshold selection strategy (clinically motivated, NOT default 0.5):

    - Diabetes:      Prioritize Sensitivity >= 0.85 (missing a diabetic is
                     costly — the patient loses the chance for early treatment).
                     Find the highest threshold where sensitivity stays >= 0.85.

    - Heart Disease: Maximize F1-score (balance precision and recall because
                     false positives trigger expensive workups while false
                     negatives miss treatable disease).

    - Liver Disease: Prioritize Specificity >= 0.80 (reduce false-alarm burden
                     — liver biopsies are invasive; screen only high-confidence
                     positives). Find the lowest threshold where specificity
                     stays >= 0.80.
    """
    y_test_arr = np.array(y_test)

    fpr, tpr, roc_thresholds = roc_curve(y_test_arr, y_prob)
    specificity = 1 - fpr
    sensitivity = tpr

    if disease_name == "Diabetes":
        # Find thresholds where sensitivity >= 0.85, pick the highest one
        # (highest threshold = most conservative, fewest false positives)
        mask = sensitivity >= 0.85
        if mask.any():
            optimal_threshold = float(roc_thresholds[mask].max())
        else:
            # Fall back to closest to 0.85 if none meets the criteria
            optimal_threshold = float(roc_thresholds[np.argmin(np.abs(sensitivity - 0.85))])

    elif disease_name == "Heart Disease":
        # CORRECTED STRATEGY: Prioritise sensitivity >= 0.75
        # Clinical rationale: Undetected heart disease risk has severe consequences.
        # Missing a true positive (false negative) is far more costly than a false
        # alarm. We accept lower specificity to ensure high-risk patients are flagged.
        # (Previous strategy was Max F1, which produced Sensitivity=0.358 — clinically
        # dangerous. Max F1 optimises the balance between precision and recall but does
        # not guarantee adequate sensitivity in imbalanced clinical settings.)
        mask = sensitivity >= 0.75
        if mask.any():
            # Among all thresholds with sensitivity >= 0.75, pick the one with
            # highest specificity (lowest false alarm rate)
            valid_idx = np.where(mask)[0]
            best_idx = valid_idx[np.argmax(specificity[valid_idx])]
            optimal_threshold = float(roc_thresholds[best_idx])
        else:
            # Fallback: 0.75 sensitivity not achievable — use max sensitivity threshold
            print(f"  ⚠️  WARNING: Sensitivity >= 0.75 not achievable — using max sensitivity threshold")
            optimal_threshold = float(roc_thresholds[np.argmax(sensitivity)])

    elif disease_name == "Liver Disease":
        # Find the lowest threshold index where specificity >= 0.80
        # (roc_curve returns fpr in ascending order → specificity descending)
        valid_idx = np.where(specificity >= 0.80)[0]
        if len(valid_idx) > 0:
            # Pick the entry with lowest FPR that still satisfies constraint
            optimal_threshold = float(roc_thresholds[valid_idx[-1]])
        else:
            optimal_threshold = 0.5

    else:
        optimal_threshold = 0.5

    # ── Compute clinical metrics at the chosen threshold ──────────────────────
    y_pred = (y_prob >= optimal_threshold).astype(int)
    tp = int(((y_pred == 1) & (y_test_arr == 1)).sum())
    tn = int(((y_pred == 0) & (y_test_arr == 0)).sum())
    fp = int(((y_pred == 1) & (y_test_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_test_arr == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    strategy_display = {
        "Diabetes":      "Sensitivity \u2265 0.85 (missing a diabetic is clinically costly)",
        "Heart Disease": "Sensitivity \u2265 0.75 \u2014 undetected heart disease risk is clinically dangerous",
        "Liver Disease": "Specificity \u2265 0.80 (reduce false-alarm burden for invasive tests)",
    }.get(disease_name, "Default 0.5")

    print(f"  Optimal Threshold : {optimal_threshold:.3f}")
    print(f"  Sensitivity       : {sens:.3f}")
    print(f"  Specificity       : {spec:.3f}")
    print(f"  PPV               : {ppv:.3f}")
    print(f"  NPV               : {npv:.3f}")

    return {
        "value":       round(optimal_threshold, 3),
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "ppv":         round(ppv, 3),
        "npv":         round(npv, 3),
        "strategy":    strategy_display,
    }


# ── Run extended metrics for each disease ────────────────────────────────────
validation_output = {}

for disease, cfg in MODELS_CONFIG.items():
    json_key = DISEASE_KEY_MAP[disease]

    # Reproduce the exact same holdout split (random_state=42, stratify)
    df_clean = df_merged.dropna(subset=[cfg["target"]]).copy()
    X        = df_clean[cfg["features"]]
    y        = df_clean[cfg["target"]]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model  = joblib.load(cfg["model_path"])
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_arr = np.array(y_test)

    ext    = compute_extended_metrics(model, X_test, y_test, disease)
    thresh = find_optimal_threshold(y_test_arr, y_prob, disease)

    validation_output[json_key] = {
        "roc_auc":      round(ext["roc_auc"], 4),
        "pr_auc":       round(ext["pr_auc"], 4),
        "brier_score":  round(ext["brier_score"], 4),
        "brier_skill":  ext["brier_skill"],
        "prevalence":   ext["prevalence"],
        "threshold":    thresh,
    }

# ── Write validation_metrics.json ────────────────────────────────────────────
OUT_JSON = os.path.join(DATA_DIR, "saved_models", "validation_metrics.json")
with open(OUT_JSON, "w") as f:
    json.dump(validation_output, f, indent=2)

print(f"\n[✓] Saved: {OUT_JSON}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"{'=' * 65}")
print("  STEP 4 SUMMARY")
print(f"{'=' * 65}")
print(f"  {'Disease':<20} {'ROC-AUC':>8} {'PR-AUC':>8} {'BSS':>8} {'Prev':>6} {'Thresh':>8} {'Sens':>6} {'Spec':>6}")
print(f"  {'─' * 72}")
for jk, v in validation_output.items():
    t = v["threshold"]
    print(f"  {jk:<20} {v['roc_auc']:>8.4f} {v['pr_auc']:>8.4f} "
          f"{v['brier_skill']:>8.4f} {v['prevalence']:>6.3f} "
          f"{t['value']:>8.3f} {t['sensitivity']:>6.3f} {t['specificity']:>6.3f}")

print(f"\n  Brier Skill Score guide:")
print(f"    >=0.25 = Acceptable probabilistic skill")
print(f"    0.10-0.25 = Weak (limited improvement over naive baseline)")
print(f"    <0.10 = WARNING: model barely beats naive prediction")
print(f"    <0.0  = Model is WORSE than a naive baseline — serious problem")
print(f"\n  (Do NOT use raw Brier Score alone for rare-disease evaluation)")
print(f"\n{'=' * 65}")
print("  Extended evaluation complete. JSON written.")
print(f"{'=' * 65}\n")
