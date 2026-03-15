"""
Random Forest Severity Classifier  —  Indian Road Accident Dataset
====================================================================
Trains a Random Forest on the clustered dataset to predict accident
severity (1=Slight, 2=Serious, 3=Fatal).  Uses cluster IDs from DBSCAN
as an input feature so the model is spatially aware.
"""

import os, sys, json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DATA_PATH, RF_MODEL_PATH, FEATURE_IMPORTANCES_PATH,
    MODELS_DIR, EDA_OUTPUT_DIR,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_TEST_SIZE,
)

# Features selected for the Indian dataset
FEATURE_COLS = [
    "Hour",
    "DayOfWeek",
    "Is_Night",
    "Weather_Binned_Enc",
    "Num_Vehicles",
    "Type_of_vehicle_Enc",
    "Road_surface_type_Enc",
    "Road_surface_conditions_Enc",
    "Light_conditions_Enc",
    "Type_of_collision_Enc",
    "Cause_of_accident_Enc",
    "Road_allignment_Enc",
    "Types_of_Junction_Enc",
    "Lanes_or_Medians_Enc",
    "Driving_experience_Enc",
    "Age_band_of_driver_Enc",
    "Cluster_ID",
]

TARGET_COL = "Severity"


def run():
    print("[CLASSIFIER] Loading processed data ...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Only train on clustered points (exclude noise)
    df = df[df["Cluster_ID"] != -1].copy()
    print(f"  Training rows (clustered only): {len(df):,}")

    # Keep only feature columns that exist in the dataframe
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  Note: missing features (skipped): {missing}")

    X = df[available_features].values
    y = df[TARGET_COL].values

    # Check class counts for stratification
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    # If any class has fewer samples than needed for stratification, adjust
    min_class_count = counts.min()
    if min_class_count < 2:
        print("  Warning: some severity classes have < 2 samples; disabling stratification.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=RF_TEST_SIZE, random_state=42, stratify=stratify,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print(f"[CLASSIFIER] Training RandomForest "
          f"(n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}) ...")
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Confusion Matrix:\n{cm}\n")
    print(classification_report(y_test, y_pred))

    # Feature importances
    importances = dict(zip(available_features, clf.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("  Feature importances:")
    for name, imp in sorted_imp:
        print(f"    {name:35s} {imp:.4f}")

    # Persist model & artefacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, RF_MODEL_PATH)
    joblib.dump(importances, FEATURE_IMPORTANCES_PATH)
    print(f"\n  Saved RF model           → {RF_MODEL_PATH}")
    print(f"  Saved feature importances → {FEATURE_IMPORTANCES_PATH}")

    # Save metrics JSON for API
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    metrics = {
        "accuracy": round(acc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                 for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
            for k, v in report.items()
        },
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": RF_MAX_DEPTH,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_columns": available_features,
        "severity_classes": {
            "1": "Slight Injury",
            "2": "Serious Injury",
            "3": "Fatal Injury",
        },
    }
    metrics_path = os.path.join(EDA_OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved model metrics → {metrics_path}")
    print("[CLASSIFIER] Done.")
    return clf, importances


if __name__ == "__main__":
    run()
