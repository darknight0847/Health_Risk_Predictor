# Model Card — MedScan AI v1.1.0

## Model Details

| Field | Value |
|---|---|
| **Developer** | Dev Patel |
| **Model date** | June 2026 |
| **Model version** | 1.1.0 |
| **Model type** | XGBoost classifiers (sklearn Pipeline) with Platt sigmoid calibration on Heart Disease model |
| **Training framework** | scikit-learn 1.3+, XGBoost 2.0+, SHAP 0.44+ |
| **Contact** | dev4447f@gmail.com |
| **License** | MIT |

---

## Intended Use

### Primary intended use
Educational screening tool demonstrating how machine learning can be applied to population-level clinical biomarker data. Intended for ML portfolio demonstration, academic discussion, and exploring XAI techniques in healthcare.

### Primary intended users
Machine learning researchers, students, and healthcare informaticists exploring ML-assisted screening concepts.

### Out-of-scope uses
**This model must not be used for:**
- Actual clinical diagnosis or treatment decisions
- Replacing physician assessment or laboratory confirmation
- Screening individuals without access to follow-up clinical care
- Any deployment in a healthcare setting, real or simulated, affecting real patients

---

## Training Data

### Source
CDC NHANES (National Health and Nutrition Examination Survey) — publicly available at [https://www.cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes)

### Critical limitation — cross-sectional design
**NHANES is a cross-sectional survey.** It measures disease *presence at a single point in time*, not disease *onset over time*. These models therefore learn to identify people who *currently have* a condition, not to predict who *will develop* it. A patient with undiagnosed diabetes for 10 years has elevated glucose *because of* their disease — not as a prospective risk factor. This is a fundamental limitation that affects clinical interpretability of all three models.

### Data size
- Total records used: 9,254
- Training set: 7,403
- Calibration set (Heart Disease only): 20% of training data (~889)
- Holdout test set: 9,254 records (stratified split)

### Disease prevalence in holdout test set
| Disease | Positive cases | Prevalence |
|---|---|---|
| Diabetes | 179 | 10.28% |
| Heart Disease | 53 | 4.77% |
| Liver Disease | 59 | 5.31% |

### Known data limitations
- NHANES oversamples certain demographic groups (elderly, low-income, minorities) by design — this may affect generalisability to other populations
- Missing values imputed using median imputation; patients with extreme missingness patterns may receive unreliable predictions
- No temporal data — the models cannot capture disease trajectory or treatment history

---

## Model Architecture & Training

### All three diseases
- **Pipeline:** `SimpleImputer(median)` → `StandardScaler()` → `XGBClassifier`
- **Hyperparameter optimisation:** Optuna
- **Explainability:** SHAP TreeExplainer (local waterfall analysis per prediction)

### Heart Disease — additional calibration step
- **Class imbalance correction:** `scale_pos_weight=19.95` (ratio of negative to positive training samples)
- **Post-hoc calibration:** Platt sigmoid scaling via `CalibratedClassifierCV(cv='prefit', method='sigmoid')`
- **Reason sigmoid over isotonic:** Calibration set contained only 42 positive samples — isotonic regression would overfit below 50 positives
- **Effect:** Mean predicted probability corrected from 19.2% → 4.78% (true prevalence: 4.77%)

---

## Evaluation Results

### Holdout test set performance (9,254 records)

| Disease | ROC-AUC | PR-AUC | Brier Score | Brier Skill Score | Threshold | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Diabetes | 0.9083 | 0.6171 | 0.0578 | 0.3727 | 0.098 | 0.860 | 0.791 |
| Heart Disease | 0.8046 | 0.1888 | 0.0429 | +0.0567 | 0.026 | 0.755 | 0.710 |
| Liver Disease | 0.7460 | 0.1712 | 0.0476 | 0.0538 | 0.070 | 0.542 | 0.802 |

### Threshold selection strategy per disease

| Disease | Strategy | Clinical rationale |
|---|---|---|
| Diabetes | Sensitivity ≥ 0.85 | Missing a diabetic is clinically costly — hyperglycaemia causes irreversible end-organ damage |
| Heart Disease | Sensitivity ≥ 0.75 | Undetected cardiovascular risk has severe acute consequences |
| Liver Disease | Specificity ≥ 0.80 | Liver disease screening in low-prevalence settings should minimise unnecessary follow-up burden |

### Metric interpretation notes
- **PR-AUC** is reported alongside ROC-AUC because all three diseases are class-imbalanced. ROC-AUC is optimistic under imbalance; PR-AUC is more conservative and informative.
- **Brier Skill Score** normalises raw Brier Score against a naive baseline (always predicting prevalence). A score of 0 means the model adds no probabilistic value; 1 is perfect. Heart Disease BSS of +0.0567 is low but positive — acceptable given 5 features and 4.77% prevalence.
- Heart Disease PR-AUC of 0.1888 should be interpreted against the random baseline of 0.048 (equal to prevalence). The model achieves ~4.0× random performance on the positive class.

---

## Known Limitations & Failure Modes

1. **Heart Disease model has only 5 features.** Clinical gold standards (Framingham Risk Score, ACC/AHA Pooled Cohort Equations) use 8–10 features including smoking status, HDL cholesterol, LDL cholesterol, and diabetes status. Performance should be interpreted accordingly.

2. **Low-probability compression in Heart Disease.** Before calibration, the model compressed most predictions toward zero due to class imbalance. Platt calibration corrected the mean but individual predictions in the 0.05–0.20 range should be treated with caution.

3. **No subgroup performance analysis.** Model performance has not been independently evaluated by age group, sex, BMI category, or ethnicity. It is possible the model performs materially differently across demographic subgroups. This analysis is planned for v1.2.

4. **Median imputation for missing values.** Patients with multiple missing biomarkers will receive predictions based on population-median substitutions — the reliability of predictions degrades as more fields are missing.

5. **Liver Disease model has the weakest discriminative performance** (AUC 0.75, PR-AUC 0.17). Predictions from this model should be treated with the greatest scepticism. Clinical liver disease screening uses FIB-4 index and NAFLD fibrosis score — this model does not replicate those validated instruments.

---

## Ethical Considerations

- **No direct harm pathway in current deployment** — the app carries an explicit educational disclaimer on every prediction
- **Risk of misuse** — the tool could be misused by individuals making health decisions without clinical guidance; the disclaimer and this model card are the primary mitigations
- **Demographic fairness** — subgroup fairness analysis (demographic parity, equalised odds across sex and age groups) has not been completed and is a known gap. Until this analysis is done, the model should not be used in any context where differential performance across groups would be consequential.
- **No PII collected** — the app does not store any user-submitted clinical data

---

## Recommendations for Future Work

- [ ] Subgroup performance breakdown by age (< 40, 40–60, 60+) and sex — v1.2 priority
- [ ] Comparison against clinical baselines: Framingham (heart), FINDRISC (diabetes), FIB-4 (liver)
- [ ] MLflow experiment tracking for all retraining runs
- [ ] Prospective validation dataset (NHANES follow-up surveys have longitudinal components)
- [ ] Fairlearn demographic parity and equalised odds analysis

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | May 2026 | Initial deployment — XGBoost + SHAP, Flask + Docker |
| 1.1.0 | June 2026 | Added PR-AUC, Brier Skill Score, calibration curves; corrected Heart Disease class imbalance with scale_pos_weight=19.95 and Platt sigmoid calibration; disease-specific clinical thresholds; REST API endpoint |
