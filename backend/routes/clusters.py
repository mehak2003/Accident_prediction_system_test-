"""API routes for spatial cluster data and GeoJSON serving."""

from flask import Blueprint, jsonify, request
import joblib
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ARI_DATA_PATH
from utils.geojson_utils import clusters_to_geojson
from utils.db import get_connection

clusters_bp = Blueprint("clusters", __name__)


def _load_clusters_from_file():
    """Fallback: load from joblib when MySQL is unavailable."""
    import numpy as np
    df = joblib.load(ARI_DATA_PATH)
    rows = df.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
        if "Pred_Severity" not in row:
            row["Pred_Severity"] = row.get("Mean_Severity", 0)
        if "Env_Modifier" not in row:
            row["Env_Modifier"] = row.get("Dominant_Weather", "")
    return rows


def _load_clusters_from_db():
    """Primary: load joined cluster + risk data from MySQL."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT
            sc.Cluster_ID, sc.Centroid_Lat, sc.Centroid_Lon,
            sc.Radius_Eps, sc.Incident_Count,
            ra.Pred_Severity, ra.ARI_Score, ra.Risk_Tier, ra.Env_Modifier
        FROM tbl_Spatial_Clusters sc
        LEFT JOIN tbl_Risk_Assessments ra ON sc.Cluster_ID = ra.Cluster_ID
        ORDER BY ra.ARI_Score DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@clusters_bp.route("/api/clusters", methods=["GET"])
def get_clusters():
    """Return all clusters as GeoJSON. Accepts ?format=json for raw list."""
    try:
        rows = _load_clusters_from_db()
    except Exception:
        rows = _load_clusters_from_file()

    fmt = request.args.get("format", "geojson")
    if fmt == "json":
        return jsonify(rows)
    return jsonify(clusters_to_geojson(rows))


@clusters_bp.route("/api/clusters/<int:cluster_id>", methods=["GET"])
def get_cluster_detail(cluster_id):
    """Return detailed info for a single cluster with its accident records."""
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT sc.*, ra.Pred_Severity, ra.ARI_Score, ra.Risk_Tier, ra.Env_Modifier
            FROM tbl_Spatial_Clusters sc
            LEFT JOIN tbl_Risk_Assessments ra ON sc.Cluster_ID = ra.Cluster_ID
            WHERE sc.Cluster_ID = %s
        """, (cluster_id,))
        cluster = cur.fetchone()
        if not cluster:
            return jsonify({"error": "Cluster not found"}), 404

        cur.execute("""
            SELECT Record_ID, Latitude, Longitude, Timestamp,
                   Weather_Cond, Severity_Hist
            FROM tbl_Accident_Records
            WHERE Cluster_ID = %s
            ORDER BY Timestamp DESC
            LIMIT 500
        """, (cluster_id,))
        accidents = cur.fetchall()

        # Stringify datetimes for JSON serialisation
        for a in accidents:
            if a.get("Timestamp"):
                a["Timestamp"] = str(a["Timestamp"])

        cur.close()
        conn.close()

        return jsonify({
            "cluster": cluster,
            "accidents": accidents,
            "accident_count": len(accidents),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
