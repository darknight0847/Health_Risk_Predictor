import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import optuna
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HealthRiskPredictor:
    def __init__(self, data_dir):
        """Initialize the predictor with the directory containing NHANES .xpt files."""
        self.data_dir = data_dir
        self.df_final = None
        self.models = {}
        self.background_data = {}
        
        # Configuration based on stability selection
        self.models_config = {
            'Diabetes': {
                'target': 'DIQ010',
                'features': ['RIDAGEYR', 'LBDGLUSI', 'LBXGLU', 'BMXWAIST', 'BMXBMI', 'BPXSY1', 'URDACT']
            },
            'Heart Disease': {
                'target': 'MCQ160C',
                'features': ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC']
            },
            'Liver Disease': {
                'target': 'MCQ160L',
                'features': ['RIDAGEYR', 'LBXGLU', 'BPXSY1', 'URXUMA', 'LBDGLUSI']
            }
        }

    def load_and_merge_data(self):
        """Loads all .xpt files and merges them on the SEQN identifier."""
        print("📂 Loading NHANES data files...")
        
        # Load Features
        df_demo = pd.read_sas(f"{self.data_dir}/DEMO_J.xpt")[['SEQN', 'RIAGENDR', 'RIDAGEYR']]
        df_glucose = pd.read_sas(f"{self.data_dir}/GLU_J.xpt")[['SEQN', 'LBXGLU', 'LBDGLUSI']]
        df_bmx = pd.read_sas(f"{self.data_dir}/BMX_J.xpt")[['SEQN', 'BMXBMI', 'BMXWAIST']]
        df_chol = pd.read_sas(f"{self.data_dir}/TCHOL_J.xpt")[['SEQN', 'LBXTC']]
        df_bp = pd.read_sas(f"{self.data_dir}/BPX_J.xpt")[['SEQN', 'BPXSY1', 'BPXDI1']]
        df_smk = pd.read_sas(f"{self.data_dir}/SMQ_J.xpt")[['SEQN', 'SMQ040']]
        df_alb = pd.read_sas(f"{self.data_dir}/ALB_CR_J.xpt")[['SEQN', 'URXUMA', 'URXUCR', 'URDACT']]
        
        # Load Labels
        df_label1 = pd.read_sas(f"{self.data_dir}/DIQ_J.xpt")[['SEQN', 'DIQ010']]
        df_label2 = pd.read_sas(f"{self.data_dir}/MCQ_J.xpt")[['SEQN', 'MCQ160C', 'MCQ160L']]
        
        dfs = [df_demo, df_glucose, df_bmx, df_chol, df_bp, df_smk, df_alb, df_label1, df_label2]
        
        # Merge all dataframes on SEQN
        self.df_final = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='left'), dfs)
        print(f"✅ Data merged successfully. Total records: {len(self.df_final)}")
        
        self._preprocess_targets()

    def _preprocess_targets(self):
        """Cleans the target variables mapping Yes=1, No=0, and Refused/Don't Know=NaN."""
        target_map = {1.0: 1, 2.0: 0, 7.0: np.nan, 9.0: np.nan, 3.0: np.nan}
        targets = ['DIQ010', 'MCQ160C', 'MCQ160L']
        
        for t in targets:
            self.df_final[t] = self.df_final[t].replace(target_map)
        print("✅ Target variables cleaned and standardized.")

    def train_models(self, n_trials=30):
        """Trains an optimized XGBoost pipeline for each disease."""
        print("\n🚀 Starting Model Training Pipeline...\n")
        
        for name, config in self.models_config.items():
            target = config['target']
            features = config['features']
            
            # 1. Isolate clean data for this specific target
            df_clean = self.df_final.dropna(subset=[target]).copy()
            X = df_clean[features]
            y = df_clean[target]
            
            # 2. Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            #SHAP analysis
            self.background_data[name] = shap.sample(X_train, 100)
            
            # 3. Dynamic Class Imbalance Calculation
            num_neg = (y_train == 0).sum()
            num_pos = (y_train == 1).sum()
            dynamic_weight = np.sqrt(num_neg / num_pos)
            
            print(f"--- Training {name} Model ---")
            print(f"Dataset: {len(X_train)} train rows | Imbalance Weight: {dynamic_weight:.2f}")

            # 4. Optuna Objective inside the loop
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'scale_pos_weight': dynamic_weight,
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
                
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('xgb', XGBClassifier(**params))
                ])
                
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict_proba(X_test)[:, 1]
                return roc_auc_score(y_test, preds)

            # 5. Run Optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            best_params['scale_pos_weight'] = dynamic_weight
            best_params['eval_metric'] = 'logloss'
            best_params['random_state'] = 42
            
            # 1. Initialize the raw XGBoost model
            base_xgb = XGBClassifier(**best_params)
            
            # 2. Wrap it in a Calibrator (Isotonic regression works best for tree models)
            calibrated_xgb = CalibratedClassifierCV(base_xgb, method='isotonic', cv=5)
            
            # 3. Build the final pipeline
            final_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', calibrated_xgb)
            ])
            
            final_pipeline.fit(X_train, y_train)
            self.models[name] = final_pipeline
            
            # Evaluate
            final_preds = final_pipeline.predict_proba(X_test)[:, 1]
            final_auc = roc_auc_score(y_test, final_preds)
            print(f"✅ Final Test ROC-AUC: {final_auc:.4f}\n")

    def predict_patient(self, patient_data_dict):
        """
        Takes a dictionary of patient data and returns risk probabilities.
        Automatically handles missing keys safely.
        """
        if not self.models:
            raise ValueError("Models are not trained yet. Call train_models() first.")
            
        print(f"🩺 Generating Risk Profile...")
        patient_df = pd.DataFrame([patient_data_dict])
        results = {}
        
        for name, config in self.models_config.items():
            features = config['features']
            
            # Ensure the dataframe has the correct columns, fill missing with NaN
            patient_features = patient_df.reindex(columns=features)
            
            # Predict
            pipeline = self.models[name]
            probability = pipeline.predict_proba(patient_features)[0][1]
            results[name] = probability
            
            print(f"{name} Risk: {probability:.1%}")
            
        return results

    def explain_patient_risk(self, patient_data_dict, persona_name="Patient"):
            """Generates SHAP Waterfall plots to explain the model's exact reasoning."""
            if not self.models:
                raise ValueError("Models are not trained yet.")
                
            patient_df = pd.DataFrame([patient_data_dict])
            
            for name, config in self.models_config.items():
                features = config['features']
                patient_features = patient_df.reindex(columns=features)
                pipeline = self.models[name]
                
                # Print the probability
                prob = pipeline.predict_proba(patient_features)[0][1]
                print(f"\n{persona_name} | {name} Risk: {prob:.1%}")
                
                # 1. Create a prediction function that outputs only the positive class probability
                predict_fn = lambda x: pipeline.predict_proba(x)[:, 1]
                
                # 2. Initialize the Explainer using the background data we saved during training
                explainer = shap.Explainer(predict_fn, self.background_data[name])
                
                # 3. Calculate SHAP values for this specific patient
                shap_values = explainer(patient_features)
                
                # 4. Plot the Waterfall graph
                plt.figure(figsize=(8, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                plt.title(f"Why {name} Risk is {prob:.1%} for {persona_name}")
                plt.tight_layout()
                plt.show()

            
# ==========================================
# Execution Example
# ==========================================
# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # 1. Initialize with the path to your folder containing the .xpt files
    DATA_DIRECTORY = r"C:\Users\Dev Patel\Desktop\jupyter projects\Health_risk_predictor"
    
    predictor = HealthRiskPredictor(data_dir=DATA_DIRECTORY)
    
    # 2. Load and merge the data
    predictor.load_and_merge_data()
    
    # 3. Train the models (set n_trials=30 for speed, 50-100 for maximum accuracy)
    predictor.train_models(n_trials=30)

    # 4. Save Models and Background Data for Flask Deployment
    import os
    import joblib

    print("\n💾 Saving models to disk for Flask app...")
    
    # Create the directory FIRST
    os.makedirs('saved_models', exist_ok=True)

    # Save each trained model pipeline
    for name, pipeline in predictor.models.items():
        safe_name = name.replace(" ", "_")
        joblib.dump(pipeline, f'saved_models/{safe_name}_model.joblib')
        print(f" -> Saved: {safe_name}_model.joblib")

    # Save background data for SHAP
    joblib.dump(predictor.background_data, 'saved_models/background_data.joblib')
    print("✅ Background data saved for SHAP!")
    
    print("\n🎉 Training complete! You can now start your Flask app.")

    # ---------------------------------------------------------
    # STRESS TESTS (Commented out for deployment phase)
    # ---------------------------------------------------------
    # stress_tests = { ... } 
    # for test_name, patient_data in stress_tests.items():
    #     predictor.explain_patient_risk(patient_data, persona_name=test_name)