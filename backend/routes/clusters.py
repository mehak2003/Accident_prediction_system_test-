"""API routes for spatial cluster data and GeoJSON serving."""

from flask import Blueprint, jsonify, request
from functools import lru_cache
import joblib
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ARI_DATA_PATH
from utils.geojson_utils import clusters_to_geojson
from utils.db import get_connection

clusters_bp = Blueprint("clusters", __name__)

# Cache cluster data in memory
_cached_clusters = None
_cache_timestamp = 0


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


def _get_cached_clusters():
    """Get clusters with caching."""
    global _cached_clusters, _cache_timestamp
    import time
    
    current_time = time.time()
    # Cache for 5 minutes
    if _cached_clusters is not None and (current_time - _cache_timestamp) < 300:
        return _cached_clusters
    
    try:
        rows = _load_clusters_from_db()
    except Exception:
        try:
            rows = _load_clusters_from_file()
        except Exception:
            return []
    
    _cached_clusters = rows
    _cache_timestamp = current_time
    return rows


@clusters_bp.route("/api/clusters", methods=["GET"])
def get_clusters():
    """Return all clusters as GeoJSON. Accepts ?format=json for raw list."""
    rows = _get_cached_clusters()
    
    if not rows:
        fmt = request.args.get("format", "geojson")
        if fmt == "json":
            return jsonify([])
        return jsonify({"type": "FeatureCollection", "features": []})

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
