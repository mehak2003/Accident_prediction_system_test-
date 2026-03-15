"""API routes for on-demand severity prediction  —  Indian dataset."""

from flask import Blueprint, jsonify, request
import joblib
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RF_MODEL_PATH, LABEL_ENCODERS_PATH, FEATURE_IMPORTANCES_PATH,
    ARI_DATA_PATH, ARI_TIERS,
)

predictions_bp = Blueprint("predictions", __name__)

_model = None
_encoders = None
_ari_clusters = None
_importances = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(RF_MODEL_PATH)
    return _model


def _get_encoders():
    global _encoders
    if _encoders is None:
        _encoders = joblib.load(LABEL_ENCODERS_PATH)
    return _encoders


def _get_ari_clusters():
    global _ari_clusters
    if _ari_clusters is None:
        _ari_clusters = joblib.load(ARI_DATA_PATH)
    return _ari_clusters


def _get_importances():
    global _importances
    if _importances is None:
        _importances = joblib.load(FEATURE_IMPORTANCES_PATH)
    return _importances


def _assign_tier(ari: float) -> str:
    for tier, (lo, hi) in ARI_TIERS.items():
        if lo <= ari < hi:
            return tier
    return "Critical"


def _safe_encode(encoders, col_name, value):
    """Encode a value using a saved LabelEncoder; return 0 for unknowns."""
    le = encoders.get(col_name)
    if le is None:
        return 0
    val = str(value).strip()
    if val in le.classes_:
        return int(le.transform([val])[0])
    return 0


WEATHER_TO_ENV_SCORE = {
    "Clear": 0.15, "Rain": 0.75, "Fog": 0.80,
    "Snow": 0.85, "Wind": 0.55, "Other": 0.50,
}


@predictions_bp.route("/api/predict", methods=["POST"])
def predict_severity():
    """
    Predict accident severity for given conditions and cluster.

    Expected JSON body:
    {
        "cluster_id": 3,
        "hour": 17,
        "day_of_week": 4,
        "is_night": 1,
        "weather": "Rain",
        "num_vehicles": 2,
        "type_of_vehicle": "Automobile",
        "road_surface_type": "Asphalt roads",
        "road_surface_conditions": "Wet or damp",
        "light_conditions": "Darkness - lights unlit",
        "type_of_collision": "Vehicle with vehicle collision",
        "cause_of_accident": "No distancing",
        "road_allignment": "Tangent road with flat terrain",
        "types_of_junction": "Y Shape",
        "lanes_or_medians": "Two-way (divided with broken lines road marking)",
        "driving_experience": "5-10yr",
        "age_band_of_driver": "18-30"
    }
    """
    data = request.get_json(force=True)
    required = ["cluster_id", "hour", "weather"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    clf = _get_model()
    encoders = _get_encoders()
    clusters = _get_ari_clusters()

    weather_enc = _safe_encode(encoders, "Weather_Binned", data["weather"])
    vehicle_enc = _safe_encode(encoders, "Type_of_vehicle",
                               data.get("type_of_vehicle", "Automobile"))
    surface_type_enc = _safe_encode(encoders, "Road_surface_type",
                                    data.get("road_surface_type", "Asphalt roads"))
    surface_cond_enc = _safe_encode(encoders, "Road_surface_conditions",
                                    data.get("road_surface_conditions", "Dry"))
    light_enc = _safe_encode(encoders, "Light_conditions",
                             data.get("light_conditions", "Daylight"))
    collision_enc = _safe_encode(encoders, "Type_of_collision",
                                 data.get("type_of_collision", "Vehicle with vehicle collision"))
    cause_enc = _safe_encode(encoders, "Cause_of_accident",
                             data.get("cause_of_accident", "No distancing"))
    road_align_enc = _safe_encode(encoders, "Road_allignment",
                                  data.get("road_allignment", "Tangent road with flat terrain"))
    junction_enc = _safe_encode(encoders, "Types_of_Junction",
                                data.get("types_of_junction", "No junction"))
    lanes_enc = _safe_encode(encoders, "Lanes_or_Medians",
                             data.get("lanes_or_medians", "Undivided Two way"))
    exp_enc = _safe_encode(encoders, "Driving_experience",
                           data.get("driving_experience", "5-10yr"))
    age_enc = _safe_encode(encoders, "Age_band_of_driver",
                           data.get("age_band_of_driver", "18-30"))

    feature_vector = np.array([[
        int(data["hour"]),
        int(data.get("day_of_week", 3)),
        int(data.get("is_night", 0)),
        weather_enc,
        int(data.get("num_vehicles", 2)),
        vehicle_enc,
        surface_type_enc,
        surface_cond_enc,
        light_enc,
        collision_enc,
        cause_enc,
        road_align_enc,
        junction_enc,
        lanes_enc,
        exp_enc,
        age_enc,
        int(data["cluster_id"]),
    ]])

    pred_severity = int(clf.predict(feature_vector)[0])
    pred_proba = clf.predict_proba(feature_vector)[0].tolist()

    severity_labels = {1: "Slight Injury", 2: "Serious Injury", 3: "Fatal Injury"}

    # Recalculate ARI for the given conditions
    cluster_id = int(data["cluster_id"])
    cluster_row = clusters[clusters["Cluster_ID"] == cluster_id]
    if cluster_row.empty:
        return jsonify({"error": f"Cluster {cluster_id} not found"}), 404

    cluster_row = cluster_row.iloc[0]
    weather_raw = data["weather"]
    env_score = WEATHER_TO_ENV_SCORE.get(weather_raw, 0.5)
    density_score = float(cluster_row.get("Density_Score",
                          cluster_row.get("Incident_Count", 1) / 500))
    sev_score = float(cluster_row.get("Severity_Score", pred_severity / 3.0))

    importances = _get_importances()
    sev_feats = {"Hour", "DayOfWeek", "Is_Night", "Cluster_ID",
                 "Cause_of_accident_Enc", "Age_band_of_driver_Enc",
                 "Driving_experience_Enc"}
    env_feats = {"Weather_Binned_Enc", "Light_conditions_Enc",
                 "Road_surface_conditions_Enc"}
    infra_feats = {"Road_surface_type_Enc", "Road_allignment_Enc",
                   "Types_of_Junction_Enc", "Lanes_or_Medians_Enc",
                   "Type_of_collision_Enc", "Type_of_vehicle_Enc", "Num_Vehicles"}

    sev_imp = sum(v for k, v in importances.items() if k in sev_feats)
    env_imp = sum(v for k, v in importances.items() if k in env_feats)
    infra_imp = sum(v for k, v in importances.items() if k in infra_feats) + 0.05
    total = sev_imp + env_imp + infra_imp or 1.0
    w1, w2, w3 = sev_imp / total, infra_imp / total, env_imp / total

    ari = w1 * sev_score + w2 * density_score + w3 * env_score
    ari = max(0.0, min(1.0, ari))

    return jsonify({
        "cluster_id": cluster_id,
        "predicted_severity": pred_severity,
        "predicted_label": severity_labels.get(pred_severity, "Unknown"),
        "severity_probabilities": {
            severity_labels.get(int(cls), str(cls)): round(p, 4)
            for cls, p in zip(clf.classes_, pred_proba)
        },
        "ari_score": round(ari, 4),
        "risk_tier": _assign_tier(ari),
        "weights": {
            "W1_severity": round(w1, 4),
            "W2_density": round(w2, 4),
            "W3_environment": round(w3, 4),
        },
        "input_conditions": {
            "weather": weather_raw,
            "hour": data["hour"],
            "env_score": round(env_score, 2),
        },
    })
