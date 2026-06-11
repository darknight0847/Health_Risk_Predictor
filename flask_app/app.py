"""
MedScan AI — Multi-Disease Risk Predictor
Flask Backend
Run: python app.py   →  http://127.0.0.1:5000
"""

import matplotlib
matplotlib.use('Agg')          # must come before pyplot import

from flask import Flask, request, render_template
import joblib, pandas as pd, numpy as np, shap
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
import io, base64, os, copy, warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Absolute paths so the app works from any working directory ────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# ── Load everything at startup (once) ────────────────────────────────────────
print("[MedScan] Loading models...")
models = {
    'Diabetes':      joblib.load(os.path.join(MODELS_DIR, 'Diabetes_model.joblib')),
    'Heart Disease': joblib.load(os.path.join(MODELS_DIR, 'Heart_Disease_model.joblib')),
    'Liver Disease': joblib.load(os.path.join(MODELS_DIR, 'Liver_Disease_model.joblib')),
}
background_data = joblib.load(os.path.join(MODELS_DIR, 'background_data.joblib'))

# Use a small subset of background samples → fast SHAP (PermutationExplainer)
bg_small = {name: shap.sample(bg, 25) for name, bg in background_data.items()}
print("[MedScan] Ready.")

# ── Feature configuration ─────────────────────────────────────────────────────
MODEL_FEATURES = {
    'Diabetes':      ['RIDAGEYR', 'LBDGLUSI', 'LBXGLU', 'BMXWAIST', 'BMXBMI', 'BPXSY1', 'URDACT'],
    'Heart Disease': ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC'],
    'Liver Disease': ['RIDAGEYR', 'LBXGLU', 'LBDGLUSI', 'BPXSY1', 'URXUMA',
                      'LBXSATSI', 'LBXSASSI', 'LBXSGTSI', 'LBXSAPSI', 'LBXSTB', 'LBXSAL'],
}

# Tuned thresholds (from holdout evaluation)
THRESHOLDS = {
    'Diabetes':      0.35,
    'Heart Disease': 0.15,
    'Liver Disease': 0.10,
}

# Human-readable labels for SHAP plots
SHORT_LABELS = {
    'RIDAGEYR':  'Age',
    'RIAGENDR':  'Gender',
    'LBXGLU':   'Glucose (mg/dL)',
    'LBDGLUSI': 'Glucose (mmol/L)',
    'BMXBMI':   'BMI',
    'BMXWAIST': 'Waist (cm)',
    'BPXSY1':   'Systolic BP',
    'URDACT':   'ACR (mg/g)',
    'LBXTC':    'Cholesterol',
    'URXUMA':   'Urine Albumin',
    'LBXSATSI': 'ALT',
    'LBXSASSI': 'AST',
    'LBXSGTSI': 'GGT',
    'LBXSAPSI': 'ALP',
    'LBXSTB':   'Bilirubin',
    'LBXSAL':   'Serum Albumin',
}

FULL_LABELS = {
    'RIDAGEYR':  'Age (years)',
    'RIAGENDR':  'Gender',
    'LBXGLU':   'Blood Glucose (mg/dL)',
    'LBDGLUSI': 'Blood Glucose (mmol/L)',
    'BMXBMI':   'BMI (kg/m²)',
    'BMXWAIST': 'Waist Circumference (cm)',
    'BPXSY1':   'Systolic Blood Pressure (mmHg)',
    'URDACT':   'Albumin-Creatinine Ratio (mg/g)',
    'LBXTC':    'Total Cholesterol (mg/dL)',
    'URXUMA':   'Urine Albumin (µg/mL)',
    'LBXSATSI': 'ALT — Alanine Aminotransferase (U/L)',
    'LBXSASSI': 'AST — Aspartate Aminotransferase (U/L)',
    'LBXSGTSI': 'GGT — Gamma-Glutamyl Transferase (U/L)',
    'LBXSAPSI': 'ALP — Alkaline Phosphatase (U/L)',
    'LBXSTB':   'Total Bilirubin (mg/dL)',
    'LBXSAL':   'Serum Albumin (g/dL)',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def risk_classify(prob, thresh):
    """Return (css_level, label, hex_color) for a probability."""
    if prob < thresh * 0.40:
        return 'low',      'Low Risk',       '#10b981'
    elif prob < thresh:
        return 'moderate', 'Moderate Risk',  '#f59e0b'
    elif prob < thresh * 2.0:
        return 'high',     'High Risk',      '#ef4444'
    else:
        return 'critical', 'Critical Risk',  '#dc2626'


def compute_shap_and_plot(model, bg, df, feats):
    """
    Returns (base64_png, pos_factors, neg_factors)
    where pos/neg_factors are lists of (label, shap_val, bar_pct).
    """
    predict_fn = lambda x, m=model: m.predict_proba(x)[:, 1]
    explainer  = shap.Explainer(predict_fn, bg)
    sv         = explainer(df)

    # Rename feature labels for the waterfall plot
    sv_renamed = shap.Explanation(
        values        = sv.values,
        base_values   = sv.base_values,
        data          = sv.data,
        feature_names = [SHORT_LABELS.get(f, f) for f in feats],
    )

    # ── Waterfall plot ────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    shap.plots.waterfall(sv_renamed[0], show=False, max_display=10)
    fig = plt.gcf()
    fig.patch.set_facecolor('#0d1b2e')
    for ax in fig.get_axes():
        ax.set_facecolor('#0d1b2e')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.spines['bottom'].set_color('#1e3a5f')
        ax.spines['left'].set_color('#1e3a5f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor='#0d1b2e', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close('all')

    # ── Top factors ───────────────────────────────────────────────────────────
    raw_vals = sv.values[0]                               # shape (n_features,)
    pairs    = [(FULL_LABELS.get(f, f), float(v)) for f, v in zip(feats, raw_vals)]

    pos_raw  = sorted([(n, v) for n, v in pairs if v > 0], key=lambda x: -x[1])[:3]
    neg_raw  = sorted([(n, v) for n, v in pairs if v < 0], key=lambda x: x[1])[:3]

    max_pos  = pos_raw[0][1] if pos_raw else 1.0
    max_neg  = abs(neg_raw[0][1]) if neg_raw else 1.0

    pos = [(n, v, min(int(v / max_pos * 100), 100))         for n, v in pos_raw]
    neg = [(n, v, min(int(abs(v) / max_neg * 100), 100))    for n, v in neg_raw]

    return b64, pos, neg


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    results   = None
    form_data = {}
    error     = None

    if request.method == 'POST':
        try:
            g = float(request.form['glucose'])
            inputs = {
                'RIDAGEYR':  float(request.form['age']),
                'RIAGENDR':  float(request.form['gender']),
                'LBXGLU':   g,
                'LBDGLUSI': round(g * 0.0555, 4),
                'BMXBMI':   float(request.form['bmi']),
                'BMXWAIST': float(request.form['waist']),
                'BPXSY1':   float(request.form['sys_bp']),
                'URDACT':   float(request.form['acr']),
                'LBXTC':    float(request.form['cholesterol']),
                'URXUMA':   float(request.form['albumin']),
                'LBXSATSI': float(request.form['alt']),
                'LBXSASSI': float(request.form['ast']),
                'LBXSGTSI': float(request.form['ggt']),
                'LBXSAPSI': float(request.form['alp']),
                'LBXSTB':   float(request.form['bilirubin']),
                'LBXSAL':   float(request.form['serum_albumin']),
            }
            form_data = request.form.to_dict()
            results   = {}

            for disease, model in models.items():
                feats  = MODEL_FEATURES[disease]
                df     = pd.DataFrame([{k: inputs[k] for k in feats}], columns=feats)
                prob   = float(model.predict_proba(df)[0][1])
                thresh = THRESHOLDS[disease]
                lvl, lbl, color = risk_classify(prob, thresh)

                plot_b64, pos, neg = compute_shap_and_plot(
                    model, bg_small[disease], df, feats
                )

                # Gauge math: circumference for r=45 is 282.74
                circ   = 282.74
                offset = round(circ * (1 - prob), 2)

                results[disease] = dict(
                    prob     = prob,
                    prob_pct = f'{prob * 100:.1f}',
                    thresh   = thresh,
                    thresh_pct = int(thresh * 100),
                    level    = lvl,
                    label    = lbl,
                    color    = color,
                    circ     = circ,
                    offset   = offset,
                    plot     = plot_b64,
                    pos      = pos,   # [(label, shap_val, bar_pct), ...]
                    neg      = neg,
                )

        except Exception as exc:
            import traceback
            error = str(exc)
            traceback.print_exc()

    return render_template('index.html',
                           results=results,
                           form_data=form_data,
                           error=error)


if __name__ == '__main__':
    # Default to port 5000 locally, but use the cloud provider's injected port (e.g. 7860 on HF)
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 so the app is accessible outside the container
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=port)
