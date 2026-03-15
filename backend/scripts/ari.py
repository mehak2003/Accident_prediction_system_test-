"""
Accident Risk Index (ARI) Computation  —  Indian Road Accident Dataset
=======================================================================
ARI = W1 * Severity_Score + W2 * Accident_Density + W3 * Environmental_Factor

Weights are derived from Random Forest feature importances.
Results are persisted to disk and optionally seeded into MySQL.
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLUSTER_DATA_PATH, FEATURE_IMPORTANCES_PATH, ARI_DATA_PATH,
    MODELS_DIR, EDA_OUTPUT_DIR, ARI_TIERS,
)

ENV_RISK_SCORES = {
    "Clear": 0.15,
    "Rain": 0.75,
    "Fog": 0.80,
    "Snow": 0.85,
    "Wind": 0.55,
    "Other": 0.50,
}

# Feature groups for weight derivation
SEVERITY_FEATURES = {
    "Hour", "DayOfWeek", "Is_Night", "Cluster_ID",
    "Cause_of_accident_Enc", "Age_band_of_driver_Enc",
    "Driving_experience_Enc",
}
ENV_FEATURES = {
    "Weather_Binned_Enc", "Light_conditions_Enc",
    "Road_surface_conditions_Enc",
}
INFRA_FEATURES = {
    "Road_surface_type_Enc", "Road_allignment_Enc",
    "Types_of_Junction_Enc", "Lanes_or_Medians_Enc",
    "Type_of_collision_Enc", "Type_of_vehicle_Enc",
    "Num_Vehicles",
}


def derive_weights(importances: dict) -> tuple:
    """Compute W1, W2, W3 from RF feature importances."""
    sev_imp = sum(v for k, v in importances.items() if k in SEVERITY_FEATURES)
    env_imp = sum(v for k, v in importances.items() if k in ENV_FEATURES)
    infra_imp = sum(v for k, v in importances.items() if k in INFRA_FEATURES)
    density_imp = infra_imp + 0.05   # base boost for density

    total = sev_imp + density_imp + env_imp
    if total == 0:
        return 0.4, 0.3, 0.3

    w1 = sev_imp / total
    w2 = density_imp / total
    w3 = env_imp / total
    return round(w1, 4), round(w2, 4), round(w3, 4)


def assign_tier(ari: float) -> str:
    for tier, (lo, hi) in ARI_TIERS.items():
        if lo <= ari < hi:
            return tier
    return "Critical"


def run():
    print("[ARI] Loading cluster data & feature importances ...")
    clusters = joblib.load(CLUSTER_DATA_PATH)
    importances = joblib.load(FEATURE_IMPORTANCES_PATH)

    w1, w2, w3 = derive_weights(importances)
    print(f"  Derived weights → W1(severity)={w1}  W2(density)={w2}  W3(env)={w3}")

    # -- Severity Score (normalised 0-1) --
    max_sev = clusters["Mean_Severity"].max()
    min_sev = clusters["Mean_Severity"].min()
    sev_range = max_sev - min_sev if max_sev != min_sev else 1.0
    clusters["Severity_Score"] = (clusters["Mean_Severity"] - min_sev) / sev_range

    # -- Accident Density (normalised 0-1) --
    max_count = clusters["Incident_Count"].max()
    clusters["Density_Score"] = clusters["Incident_Count"] / max_count if max_count else 0

    # -- Environmental Factor --
    clusters["Env_Score"] = clusters["Dominant_Weather"].map(ENV_RISK_SCORES).fillna(0.5)

    # -- ARI --
    clusters["ARI_Score"] = (
        w1 * clusters["Severity_Score"] +
        w2 * clusters["Density_Score"] +
        w3 * clusters["Env_Score"]
    )
    clusters["ARI_Score"] = clusters["ARI_Score"].clip(0.0, 1.0)
    clusters["Risk_Tier"] = clusters["ARI_Score"].apply(assign_tier)

    # Summary
    tier_counts = clusters["Risk_Tier"].value_counts()
    print(f"\n  Risk Tier distribution:")
    for tier, cnt in tier_counts.items():
        print(f"    {tier:10s} : {cnt}")
    print(f"\n  ARI range : [{clusters['ARI_Score'].min():.4f} , {clusters['ARI_Score'].max():.4f}]")
    print(clusters[["Cluster_ID", "Centroid_Lat", "Centroid_Lon",
                     "Incident_Count", "ARI_Score", "Risk_Tier"]].head(10).to_string(index=False))

    # Persist
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clusters, ARI_DATA_PATH)
    print(f"\n  Saved ARI data → {ARI_DATA_PATH}")

    # Save JSON for API
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    ari_json = clusters.to_dict(orient="records")
    for row in ari_json:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = round(float(v), 6)
    ari_path = os.path.join(EDA_OUTPUT_DIR, "ari_results.json")
    with open(ari_path, "w") as f:
        json.dump(ari_json, f, indent=2)
    print(f"  Saved ARI JSON  → {ari_path}")
    print("[ARI] Done.")
    return clusters


if __name__ == "__main__":
    run()
