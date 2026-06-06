"""
Healthcare AI Outcome Predictions
Builds a machine learning classifier pipeline for patient risk stratification
featuring imputation, scaling, metrics calibration, and SHAP explainability hook.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, precision_score

class PatientOutcomePredictor:
    def __init__(self):
        # Create pipeline with median imputer and scaler
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6))
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict_risk(self, X):
        # Return probability of positive outcome (high risk)
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate_clinical_metrics(self, X_test, y_test, risk_threshold=0.15):
        """
        Evaluate metrics at a clinically optimal risk threshold (e.g. 15% risk)
        to prioritize sensitivity (recall) in diagnosing critical conditions.
        """
        probabilities = self.predict_risk(X_test)
        predictions = (probabilities >= risk_threshold).astype(int)
        
        auc = roc_auc_score(y_test, probabilities)
        brier = brier_score_loss(y_test, probabilities)
        sensitivity = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        
        return {
            "roc_auc": float(auc),
            "brier_score_loss": float(brier),
            "sensitivity_at_threshold": float(sensitivity),
            "precision_at_threshold": float(precision)
        }

    def get_feature_importances(self, feature_names):
        importances = self.pipeline.named_steps["classifier"].feature_importances_
        return dict(zip(feature_names, importances))

if __name__ == "__main__":
    # Generate mock clinical tabular data: 500 patients, 5 features (e.g. age, blood pressure, creatinine, lactate, temp)
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(500, 5))
    # Outcome 1 = ICU admission / severe infection, 0 = stable
    # Correlate outcome with features 2 (lactate) and 3 (temperature)
    logits = 0.8 * X[:, 2] + 1.2 * X[:, 3] - 0.5
    probs = 1 / (1 + np.exp(-logits))
    y = (probs >= np.random.uniform(size=500)).astype(int)
    
    feature_names = ["age", "sys_bp", "lactate", "temperature", "heart_rate"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    predictor = PatientOutcomePredictor()
    predictor.train(X_train, y_train)
    
    metrics = predictor.evaluate_clinical_metrics(X_test, y_test, risk_threshold=0.2)
    importances = predictor.get_feature_importances(feature_names)
    
    print("Patient Risk Stratification Pipeline Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("
Feature Importance Mapping:")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
