"""
generate_calibration_plots.py
==============================
Run this script ONCE to generate calibration curve PNGs (reliability diagrams).
Output goes to: flask_app/static/calibration/

A calibration curve checks whether predicted probabilities match actual outcome
rates. A well-calibrated model predicting 0.70 should be correct ~70% of the
time. The dashed diagonal line represents perfect calibration.

Usage (from project root):
    python flask_app/generate_calibration_plots.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import warnings

from functools import reduce

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required for server/script use
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
# This file lives at flask_app/generate_calibration_plots.py
# Project root is one level up.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'saved_models')
DATA_DIR     = PROJECT_ROOT   # NHANES .xpt files live in the project root
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, 'static', 'calibration')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model & feature config (mirrors test_holdout_evaluation.py) ───────────────
MODELS_CONFIG = {
    'Diabetes': {
        'model_path': os.path.join(MODELS_DIR, 'Diabetes_model.joblib'),
        'target':     'DIQ010',
        'features':   ['RIDAGEYR', 'LBDGLUSI', 'LBXGLU', 'BMXWAIST',
                       'BMXBMI', 'BPXSY1', 'URDACT'],
    },
    'Heart_Disease': {
        'model_path': os.path.join(MODELS_DIR, 'Heart_Disease_model.joblib'),
        'target':     'MCQ160C',
        'features':   ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC'],
    },
    'Liver_Disease': {
        'model_path': os.path.join(MODELS_DIR, 'Liver_Disease_model.joblib'),
        'target':     'MCQ160L',
        'features':   [
            'RIDAGEYR', 'LBXGLU', 'LBDGLUSI', 'BPXSY1', 'URXUMA',
            'LBXSATSI', 'LBXSASSI', 'LBXSGTSI', 'LBXSAPSI', 'LBXSTB', 'LBXSAL',
        ],
    },
}

# ── Load & merge NHANES data (same as training / test_holdout_evaluation.py) ──
print('Loading NHANES data...')

df_demo    = pd.read_sas(f'{DATA_DIR}/DEMO_J.xpt')[['SEQN', 'RIAGENDR', 'RIDAGEYR']]
df_glucose = pd.read_sas(f'{DATA_DIR}/GLU_J.xpt')[['SEQN', 'LBXGLU', 'LBDGLUSI']]
df_bmx     = pd.read_sas(f'{DATA_DIR}/BMX_J.xpt')[['SEQN', 'BMXBMI', 'BMXWAIST']]
df_chol    = pd.read_sas(f'{DATA_DIR}/TCHOL_J.xpt')[['SEQN', 'LBXTC']]
df_bp      = pd.read_sas(f'{DATA_DIR}/BPX_J.xpt')[['SEQN', 'BPXSY1', 'BPXDI1']]
df_smk     = pd.read_sas(f'{DATA_DIR}/SMQ_J.xpt')[['SEQN', 'SMQ040']]
df_alb     = pd.read_sas(f'{DATA_DIR}/ALB_CR_J.xpt')[['SEQN', 'URXUMA', 'URXUCR', 'URDACT']]
df_bio     = pd.read_sas(f'{DATA_DIR}/BIOPRO_J.xpt')[[
    'SEQN', 'LBXSATSI', 'LBXSASSI', 'LBXSGTSI', 'LBXSAPSI', 'LBXSTB', 'LBXSAL'
]]
df_label1  = pd.read_sas(f'{DATA_DIR}/DIQ_J.xpt')[['SEQN', 'DIQ010']]
df_label2  = pd.read_sas(f'{DATA_DIR}/MCQ_J.xpt')[['SEQN', 'MCQ160C', 'MCQ160L']]

dfs = [df_demo, df_glucose, df_bmx, df_chol, df_bp, df_smk,
       df_alb, df_bio, df_label1, df_label2]
df_merged = reduce(lambda l, r: pd.merge(l, r, on='SEQN', how='left'), dfs)

# Clean targets — same mapping as training
target_map = {1.0: 1, 2.0: 0, 7.0: np.nan, 9.0: np.nan, 3.0: np.nan}
for t in ['DIQ010', 'MCQ160C', 'MCQ160L']:
    df_merged[t] = df_merged[t].replace(target_map)

print(f'Merged dataset: {len(df_merged):,} records\n')

# ── Plot style constants ──────────────────────────────────────────────────────
PLOT_BG     = '#05091a'
GRID_COLOR  = '#1e3a5f'
TEXT_COLOR  = '#94a3b8'
TITLE_COLOR = '#e2e8f0'
LINE_COLOR  = '#4F46E5'   # indigo — model curve
FILL_COLOR  = '#4F46E5'
IDEAL_COLOR = '#64748b'   # muted — perfect calibration line

# Disease display names
DISPLAY_NAMES = {
    'Diabetes':      'Diabetes',
    'Heart_Disease': 'Heart Disease',
    'Liver_Disease': 'Liver Disease',
}

# ── Generate calibration plot for each disease ─────────────────────────────────
for disease_key, cfg in MODELS_CONFIG.items():
    display = DISPLAY_NAMES[disease_key]
    print(f'Generating calibration plot: {display}...')

    # Exact same holdout split as training (random_state=42, stratify)
    df_clean = df_merged.dropna(subset=[cfg['target']]).copy()
    X        = df_clean[cfg['features']]
    y        = df_clean[cfg['target']]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model  = joblib.load(cfg['model_path'])
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_arr = np.array(y_test)

    # Calibration curve — 10 uniform bins
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test_arr, y_prob, n_bins=10, strategy='uniform'
    )

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1],
            linestyle='--', color=IDEAL_COLOR, linewidth=1.4,
            label='Perfect calibration', zorder=2)

    # Model calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives,
            marker='o', markersize=7, linewidth=2.2,
            color=LINE_COLOR, label=display, zorder=3)

    # Shaded gap between model and perfect calibration
    ax.fill_between(
        mean_predicted_value,
        fraction_of_positives,
        mean_predicted_value,          # perfect calibration at same x
        alpha=0.15, color=FILL_COLOR, zorder=1
    )

    # Styling
    ax.set_xlabel('Mean Predicted Probability', fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel('Fraction of Positives',      fontsize=11, color=TEXT_COLOR)
    ax.set_title(f'Calibration Curve — {display}',
                 fontsize=12, color=TITLE_COLOR, pad=12)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    ax.grid(True, color=GRID_COLOR, alpha=0.5, linestyle='--', linewidth=0.7)

    leg = ax.legend(loc='upper left', fontsize=9,
                    facecolor='#0d1b2e', edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR)

    plt.tight_layout(pad=1.4)

    out_path = os.path.join(OUTPUT_DIR, f'{disease_key}_calibration.png')
    plt.savefig(out_path, dpi=150, facecolor=PLOT_BG, edgecolor='none')
    plt.close()
    print(f'  [✓] Saved: {out_path}')

print(f'\nAll calibration plots saved to: {OUTPUT_DIR}')
