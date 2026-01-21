# train_model.py
"""
VSCode-friendly train script with tolerant column name detection.
Place your CSV under: ./data/students_dataset_2000_balanced_50_50.csv
Run: python train_model.py

Improvements for High Confidence Accuracy:
- Enhanced hyperparameter tuning (50 iterations vs 25)
- Better scaling and feature normalization
- StratifiedKFold cross-validation (5 splits vs 3)
- Comprehensive metrics: accuracy + ROC-AUC
- NO deprecated 'early_stopping_rounds' parameter
- min_child_weight parameter added for better regularization
"""

import os
import pandas as pd
import numpy as np
import difflib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# AUTO-DETECT PROJECT BASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "students_dataset_2000_balanced_50_50.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print(">> Base directory:", BASE_DIR)
print(">> Looking for dataset at:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"\n Dataset not found!\nExpected at: {DATA_PATH}")

# LOAD DATASET
print("\n Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("OK Dataset loaded. Columns found:")
print(", ".join(list(df.columns)))
print()

# EXPECTED FIELDS & ALIASES
ALIASES = {
    'kan': ['kan','kannada','kan_marks','kannada_marks'],
    'english': ['english','english_marks','eng','eng_marks'],
    'maths': ['maths','math','math_marks','mathematics'],
    'chem': ['chem','chemistry','chemistry_marks','chem_marks'],
    'bio_or_cs': ['bio_or_cs','bio','biology','biology_or_cs','bio_marks','cs','cs_marks'],
    'physics': ['physics','physics_marks','phys','phys_marks'],
    'iq': ['iq','iq_score','iq_marks','iq_test_marks','iq_test'],
    'study_hours_per_day': ['study_hours_per_day','study_hours','study_hours_per_week'],
    'time_extra': ['time_extra','time_spent_on_extra_activity_hrs','time_spent'],
    'courses': ['courses','courses_count','extra_courses','num_courses'],
    'attendance': ['attendance','attendance_percentage','attendence'],
    'extra_activity': ['extra_activity','extra','has_extra_activity','extracurricular'],
    'attended_academic': ['attended_academic','attended_contest','attended_activity'],
    'result': ['result','outcome','label','pass_fail','final_result','status']
}

def normalize_name(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def find_column_for(alias_list, df_columns, use_fuzzy=True):
    for a in alias_list:
        for col in df_columns:
            if str(col).strip().lower() == a.strip().lower():
                return col
    alias_norms = [normalize_name(a) for a in alias_list]
    for col in df_columns:
        if normalize_name(col) in alias_norms:
            return col
    if use_fuzzy:
        norm_to_orig = {normalize_name(col): col for col in df_columns}
        choices = list(norm_to_orig.keys())
        for a in alias_list:
            a_norm = normalize_name(a)
            matches = difflib.get_close_matches(a_norm, choices, n=2, cutoff=0.75)
            if matches:
                return norm_to_orig[matches[0]]
    return None

# MAP COLUMNS
mapped = {}
missing = []
for logical_name, alias_list in ALIASES.items():
    col = find_column_for(alias_list, df.columns)
    if col:
        mapped[logical_name] = col
    else:
        missing.append((logical_name, alias_list))

if missing:
    print("\n Could not auto-map the following required logical columns:")
    for name, aliases in missing:
        print(f" - {name}")
    raise ValueError("Please rename your CSV columns to match expected fields.")

print("OK Column mapping:")
for k, v in mapped.items():
    print(f"  {k}  <--  {v}")

# Build usable DataFrame
rename_map = {mapped[k]: k for k in mapped.keys()}
df2 = df.rename(columns=rename_map)

if 'result' not in df2.columns:
    raise ValueError("Label column 'result' was not found or mapped.")

# Prepare data
numeric_cols = ['kan','english','maths','chem','bio_or_cs','physics','iq','study_hours_per_day','time_extra','courses','attendance']
for c in numeric_cols:
    if c not in df2.columns:
        df2[c] = 0
    df2[c] = pd.to_numeric(df2[c], errors='coerce')
    if df2[c].isna().all():
        df2[c] = 0.0
    else:
        df2[c] = df2[c].fillna(df2[c].mean())

binary_cols = ['extra_activity','attended_academic']
for c in binary_cols:
    if c not in df2.columns:
        df2[c] = 0
    df2[c] = df2[c].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ('1','yes','true','y','t') else 0)

# Normalize result to 0/1
if df2['result'].dtype == object or not np.issubdtype(df2['result'].dtype, np.integer):
    df2['result'] = df2['result'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ('pass','1','true','yes','p') else 0)
df2['result'] = df2['result'].fillna(0).astype(int)

# Feature matrix X and label y
feature_cols = ['kan','english','maths','chem','bio_or_cs','physics','iq','study_hours_per_day','attendance','extra_activity','time_extra','attended_academic','courses']
for c in feature_cols:
    if c not in df2.columns:
        df2[c] = 0

X = df2[feature_cols].astype(float)
y = df2['result'].astype(int)

# Train/test split with stratification
print("\n>> Splitting dataset (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scaling
print(">> Scaling numeric features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Training with enhanced RandomizedSearchCV
print("\n>> Starting XGBoost training with hyperparameter optimization...")
xgb_base = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    n_jobs=4,
    use_label_encoder=False,
    verbosity=0
)

# Enhanced parameter distribution
param_dist = {
    'n_estimators': randint(300, 700),
    'max_depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.25),
    'subsample': uniform(0.65, 0.35),
    'colsample_bytree': uniform(0.65, 0.35),
    'gamma': uniform(0.0, 3.0),
    'reg_alpha': uniform(0.0, 1.0),
    'reg_lambda': uniform(0.5, 2.0),
    'min_child_weight': randint(1, 5)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=50,  # Increased from 25 for better optimization
    scoring='accuracy',
    cv=cv,
    random_state=42,
    verbose=2,
    n_jobs=4
)

print("\n>> Fitting hyperparameter search (this may take a few minutes)...")
search.fit(X_train_scaled, y_train)

best_model = search.best_estimator_
print("\n== Best XGBoost parameters found:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")
print(f"  Best CV Score: {search.best_score_:.4f}")

# Final training
print("\n>> Final training on full training set...")
best_model.fit(X_train_scaled, y_train, verbose=False)

# Test model
print("\n>> Testing model on validation set...")
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"** ROC-AUC: {roc_auc:.4f}")
except:
    print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n-- Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

print("\n-- Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n>> Saved model to: {model_path}")
print(f">> Saved scaler to: {scaler_path}")

# Save feature columns for reference
feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
import json
with open(feature_cols_path, 'w') as f:
    json.dump(feature_cols, f)
print(f">> Saved feature columns to: {feature_cols_path}")

print("\nOK Training complete! Model saved successfully.")
print(f"\n** Final Model Performance: {accuracy*100:.2f}% accuracy on test set")
