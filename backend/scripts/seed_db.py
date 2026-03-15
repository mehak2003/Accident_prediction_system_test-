"""
Database Seeder  —  Indian Road Accident Dataset
==================================================
Populates MySQL tables with processed cluster data, accident records, and
ARI risk assessments.  Should be run after the full ML pipeline completes.
"""

import os, sys
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DATA_PATH, ARI_DATA_PATH
from utils.db import get_connection, init_database, truncate_tables


BATCH_SIZE = 5000


def seed_clusters(cur, clusters_df):
    print("[SEED] Inserting spatial clusters ...")
    sql = """
        INSERT INTO tbl_Spatial_Clusters
            (Cluster_ID, Centroid_Lat, Centroid_Lon, Radius_Eps, Incident_Count)
        VALUES (%s, %s, %s, %s, %s)
    """
    rows = []
    for _, r in clusters_df.iterrows():
        rows.append((
            int(r["Cluster_ID"]),
            float(r["Centroid_Lat"]),
            float(r["Centroid_Lon"]),
            float(r["Radius_Eps"]),
            int(r["Incident_Count"]),
        ))
    cur.executemany(sql, rows)
    print(f"  Inserted {len(rows)} clusters.")


def seed_accidents(cur, df):
    print("[SEED] Inserting accident records (batched) ...")
    df_clustered = df[df["Cluster_ID"] != -1].copy()
    sql = """
        INSERT INTO tbl_Accident_Records
            (Latitude, Longitude, Timestamp, Weather_Cond, Severity_Hist, Cluster_ID)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    total = 0
    batch = []
    for _, r in df_clustered.iterrows():
        batch.append((
            float(r["Latitude"]),
            float(r["Longitude"]),
            "2020-01-01 00:00:00",   # Placeholder — Indian dataset lacks full timestamps
            str(r.get("Weather_Binned", "Clear")),
            int(r["Severity"]),
            int(r["Cluster_ID"]),
        ))
        if len(batch) >= BATCH_SIZE:
            cur.executemany(sql, batch)
            total += len(batch)
            batch = []
    if batch:
        cur.executemany(sql, batch)
        total += len(batch)
    print(f"  Inserted {total:,} accident records.")


def seed_risk_assessments(cur, ari_df):
    print("[SEED] Inserting risk assessments ...")
    sql = """
        INSERT INTO tbl_Risk_Assessments
            (Cluster_ID, Pred_Severity, ARI_Score, Risk_Tier, Env_Modifier)
        VALUES (%s, %s, %s, %s, %s)
    """
    rows = []
    for _, r in ari_df.iterrows():
        rows.append((
            int(r["Cluster_ID"]),
            float(r.get("Mean_Severity", r.get("Severity_Score", 0))),
            float(r["ARI_Score"]),
            str(r["Risk_Tier"]),
            str(r.get("Dominant_Weather", "")),
        ))
    cur.executemany(sql, rows)
    print(f"  Inserted {len(rows)} risk assessments.")


def run():
    print("[SEED] Initialising database ...")
    init_database()
    truncate_tables()

    ari_df = joblib.load(ARI_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    conn = get_connection()
    cur = conn.cursor()
    try:
        seed_clusters(cur, ari_df)
        seed_accidents(cur, df)
        seed_risk_assessments(cur, ari_df)
        conn.commit()
        print("[SEED] All data committed to MySQL.")
    except Exception as exc:
        conn.rollback()
        print(f"[SEED] ERROR – rolled back: {exc}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run()
