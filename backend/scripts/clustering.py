"""
DBSCAN Spatial Clustering  —  Indian Road Accident Dataset
============================================================
Clusters geocoded accident coordinates using DBSCAN with haversine
distance to identify geographic "Black Spots."  Outlier/noise points
(label = -1) are filtered out.  Cluster centroids and metadata are persisted.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DATA_PATH, DBSCAN_MODEL_PATH, CLUSTER_DATA_PATH,
    MODELS_DIR, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
)


def run():
    print("[CLUSTERING] Loading processed data ...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"  {len(df):,} rows loaded.")

    coords = df[["Latitude", "Longitude"]].values

    # Convert degrees → radians for haversine metric
    coords_rad = np.radians(coords)

    print(f"[CLUSTERING] Running DBSCAN  eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES} ...")
    db = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    labels = db.fit_predict(coords_rad)
    df["Cluster_ID"] = labels

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise:,}")

    if n_clusters == 0:
        print("  WARNING: No clusters found. Try decreasing DBSCAN_EPS or DBSCAN_MIN_SAMPLES.")

    # Build cluster summary
    clustered = df[df["Cluster_ID"] != -1].copy()
    cluster_summary = (
        clustered.groupby("Cluster_ID")
        .agg(
            Centroid_Lat=("Latitude", "mean"),
            Centroid_Lon=("Longitude", "mean"),
            Incident_Count=("Latitude", "size"),
            Mean_Severity=("Severity", "mean"),
            Dominant_Weather=("Weather_Binned",
                              lambda x: x.mode().iloc[0] if len(x.mode()) else "Clear"),
            Dominant_Area=("Area_accident_occured",
                           lambda x: x.mode().iloc[0] if "Area_accident_occured" in x.index or len(x.mode()) else "Unknown")
                          if "Area_accident_occured" in clustered.columns
                          else ("Weather_Binned", lambda x: "Unknown"),
        )
        .reset_index()
    )
    cluster_summary["Radius_Eps"] = round(DBSCAN_EPS * 6371.0, 2)  # convert rad to km

    print(f"\n  Cluster summary ({len(cluster_summary)} clusters):")
    print(cluster_summary[["Cluster_ID", "Centroid_Lat", "Centroid_Lon",
                           "Incident_Count", "Mean_Severity"]].head(10).to_string(index=False))

    # Persist
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(labels, DBSCAN_MODEL_PATH)
    joblib.dump(cluster_summary, CLUSTER_DATA_PATH)
    print(f"\n  Saved cluster labels → {DBSCAN_MODEL_PATH}")
    print(f"  Saved cluster data   → {CLUSTER_DATA_PATH}")

    # Write back to processed CSV (add Cluster_ID column)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Updated {PROCESSED_DATA_PATH} with Cluster_ID column.")
    print("[CLUSTERING] Done.")
    return cluster_summary


if __name__ == "__main__":
    run()
