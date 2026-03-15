"""API routes for serving pre-computed EDA statistics  —  Indian dataset."""

from flask import Blueprint, jsonify
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EDA_OUTPUT_DIR

eda_bp = Blueprint("eda", __name__)


def _load_json(name: str):
    path = os.path.join(EDA_OUTPUT_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@eda_bp.route("/api/eda/hourly")
def eda_hourly():
    data = _load_json("hourly")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/weekly")
def eda_weekly():
    data = _load_json("weekly")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/severity")
def eda_severity():
    data = _load_json("severity")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/weather")
def eda_weather():
    data = _load_json("weather")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/top_areas")
def eda_top_areas():
    data = _load_json("top_areas")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/collision_types")
def eda_collision_types():
    data = _load_json("collision_types")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/causes")
def eda_causes():
    data = _load_json("causes")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/vehicle_types")
def eda_vehicle_types():
    data = _load_json("vehicle_types")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/severity_by_weather")
def eda_severity_by_weather():
    data = _load_json("severity_by_weather")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/severity_by_light")
def eda_severity_by_light():
    data = _load_json("severity_by_light")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/eda/summary")
def eda_summary():
    data = _load_json("summary")
    if data is None:
        return jsonify({"error": "EDA not generated yet"}), 404
    return jsonify(data)


@eda_bp.route("/api/model/metrics")
def model_metrics():
    data = _load_json("model_metrics")
    if data is None:
        return jsonify({"error": "Model not trained yet"}), 404
    return jsonify(data)
