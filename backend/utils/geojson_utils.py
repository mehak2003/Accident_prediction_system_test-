"""Utilities for converting cluster / risk data into GeoJSON FeatureCollections."""

from geojson import Feature, FeatureCollection, Point


def clusters_to_geojson(cluster_rows: list[dict]) -> dict:
    """
    Convert a list of cluster dicts into a GeoJSON FeatureCollection.

    Each dict must have at minimum:
        Centroid_Lat, Centroid_Lon, Cluster_ID, Incident_Count,
        ARI_Score, Risk_Tier
    """
    features = []
    for row in cluster_rows:
        props = {
            "cluster_id": row["Cluster_ID"],
            "incident_count": row["Incident_Count"],
            "ari_score": round(float(row.get("ARI_Score", 0)), 4),
            "risk_tier": row.get("Risk_Tier", "Unknown"),
            "pred_severity": round(float(row.get("Pred_Severity", 0)), 4),
            "env_modifier": row.get("Env_Modifier", None),
            "radius_eps": float(row.get("Radius_Eps", 0)),
        }
        point = Point((float(row["Centroid_Lon"]), float(row["Centroid_Lat"])))
        features.append(Feature(geometry=point, properties=props))
    return FeatureCollection(features)


def accidents_to_geojson(accident_rows: list[dict]) -> dict:
    """Convert individual accident records into GeoJSON points."""
    features = []
    for row in accident_rows:
        props = {
            "record_id": row.get("Record_ID"),
            "severity": row.get("Severity_Hist"),
            "weather": row.get("Weather_Cond"),
            "timestamp": str(row.get("Timestamp", "")),
            "cluster_id": row.get("Cluster_ID"),
        }
        point = Point((float(row["Longitude"]), float(row["Latitude"])))
        features.append(Feature(geometry=point, properties=props))
    return FeatureCollection(features)
