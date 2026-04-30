"""API routes for serving pre-computed EDA statistics  —  Indian dataset."""

from flask import Blueprint, jsonify
import json, os, sys
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EDA_OUTPUT_DIR

eda_bp = Blueprint("eda", __name__)

# Cache JSON files in memory
@lru_cache(maxsize=32)
def _load_json(name: str):
    path = os.path.join(EDA_OUTPUT_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@eda_bp.route("/api/eda/hourly")
def eda_hourly():
    return jsonify(_load_json("hourly"))


@eda_bp.route("/api/eda/weekly")
def eda_weekly():
    return jsonify(_load_json("weekly"))


@eda_bp.route("/api/eda/severity")
def eda_severity():
    return jsonify(_load_json("severity"))


@eda_bp.route("/api/eda/weather")
def eda_weather():
    return jsonify(_load_json("weather"))


@eda_bp.route("/api/eda/top_areas")
def eda_top_areas():
    return jsonify(_load_json("top_areas"))


@eda_bp.route("/api/eda/collision_types")
def eda_collision_types():
    return jsonify(_load_json("collision_types"))


@eda_bp.route("/api/eda/causes")
def eda_causes():
    return jsonify(_load_json("causes"))


@eda_bp.route("/api/eda/vehicle_types")
def eda_vehicle_types():
    return jsonify(_load_json("vehicle_types"))


@eda_bp.route("/api/eda/severity_by_weather")
def eda_severity_by_weather():
    return jsonify(_load_json("severity_by_weather"))


@eda_bp.route("/api/eda/severity_by_light")
def eda_severity_by_light():
    return jsonify(_load_json("severity_by_light"))


@eda_bp.route("/api/eda/summary")
def eda_summary():
    return jsonify(_load_json("summary"))


@eda_bp.route("/api/model/metrics")
def model_metrics():
    return jsonify(_load_json("model_metrics"))
