# train_model.py
"""
Train XGBoost classifier for student Pass/Fail prediction.
Assumes the CSV dataset is at: data/students_dataset_2000_balanced_50_50.csv

Output (saved in ./model):
 - xgb_model.pkl
 - scaler.pkl
 - feature_cols.json
 - metrics.txt
 - feature_importance.csv
"""
import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATA_PATH = "students_dataset_2000_balanced_50_50.csv"

MODEL_DIR = "model"
RANDOM_STATE = 42
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
print("Loading dataset:", DATA_PATH)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset rows: {len(df)}, cols: {len(df.columns)}")

# ---------- DEFINE FEATURES ----------
# Note: 'hindi' removed.
# Subject columns in CSV:
subject_cols = [
    "kan_marks",
    "english_marks",
    "maths_marks",
    "chemistry_marks",
    "biology_or_cs_marks",
    "physics_marks"
]

# Additional features expected by the app & dataset:
other_cols = [
    "iq_test_marks",
    "study_hours_per_day",
    "extra_activity",
    "time_spent_on_extra_activity_hrs",
    "attended_academic_activity",
    "courses_count",
    "attendance_percentage"
]

feature_cols = subject_cols + other_cols

required_cols = set(feature_cols) | {"result"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in dataset: {missing}")

# ---------- PREPROCESS ----------
data = df.copy()

# Normalize Yes/No columns to 1/0
yn_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
for col in ["extra_activity", "attended_academic_activity"]:
    if col in data.columns:
        data[col] = data[col].map(yn_map).fillna(0).astype(int)

# Drop rows with missing required features (alternatively impute)
data = data.dropna(subset=feature_cols + ["result"]).reset_index(drop=True)
print(f"Rows after dropping NaNs: {len(data)}")

# Map target
data["target"] = data["result"].map({"Pass": 1, "Fail": 0})
if data["target"].isnull().any():
    raise ValueError("Found 'result' values that are not 'Pass' or 'Fail'")

X = data[feature_cols].astype(float)
y = data["target"].astype(int)

print("X shape:", X.shape)
print("Target distribution (Fail, Pass):", np.bincount(y))

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# ---------- SCALE ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and feature order
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)
print("Saved scaler and feature_cols.json")

# ---------- TRAIN XGBOOST ----------
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Small grid; expand if you want longer tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(xgb_clf, param_grid, cv=cv, scoring="accuracy", verbose=2, n_jobs=2)
print("Starting GridSearchCV ...")
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# ---------- EVALUATE ----------
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])
cm = confusion_matrix(y_test, y_pred)

print(f"Test accuracy: {acc:.4f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# ---------- SAVE ARTIFACTS ----------
model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
joblib.dump(best_model, model_path)
print("Saved model:", model_path)

metrics_path = os.path.join(MODEL_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Best params: {grid.best_params_}\n")
    f.write(f"Test accuracy: {acc:.4f}\n\n")
    f.write("Confusion matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification report:\n")
    f.write(report + "\n")
print("Saved metrics to:", metrics_path)

# ---------- FEATURE IMPORTANCE ----------
fi = best_model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values("importance", ascending=False)
fi_csv = os.path.join(MODEL_DIR, "feature_importance.csv")
fi_df.to_csv(fi_csv, index=False)
print("Saved feature importance to:", fi_csv)
print(fi_df)

print("Training complete.")
