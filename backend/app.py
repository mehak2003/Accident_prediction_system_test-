"""
Flask Application — AI-Based Accident Hotspot Prediction System (India)
========================================================================
Entry point for the REST API server.  Registers all route blueprints
and configures CORS for frontend consumption.
"""

import os, sys
from flask import Flask, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.clusters import clusters_bp
from routes.predictions import predictions_bp
from routes.eda_routes import eda_bp
from routes.upload import upload_bp


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(clusters_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(upload_bp)

    @app.route("/")
    def index():
        return jsonify({
            "service": "Accident Hotspot Prediction API (India)",
            "version": "1.0.0",
            "dataset": "Road Accident Severity in India (Kaggle)",
            "endpoints": {
                "clusters_geojson":    "GET  /api/clusters",
                "cluster_detail":      "GET  /api/clusters/<id>",
                "predict":             "POST /api/predict",
                "eda_hourly":          "GET  /api/eda/hourly",
                "eda_weekly":          "GET  /api/eda/weekly",
                "eda_severity":        "GET  /api/eda/severity",
                "eda_weather":         "GET  /api/eda/weather",
                "eda_top_areas":       "GET  /api/eda/top_areas",
                "eda_collision_types": "GET  /api/eda/collision_types",
                "eda_causes":          "GET  /api/eda/causes",
                "eda_vehicle_types":   "GET  /api/eda/vehicle_types",
                "eda_sev_weather":     "GET  /api/eda/severity_by_weather",
                "eda_sev_light":       "GET  /api/eda/severity_by_light",
                "eda_summary":         "GET  /api/eda/summary",
                "model_metrics":       "GET  /api/model/metrics",
                "upload_csv":          "POST /api/upload",
                "run_pipeline":        "POST /api/pipeline/run",
                "health":              "GET  /api/health",
            },
        })

    @app.route("/api/health")
    def health():
        from config import (
            RF_MODEL_PATH, ARI_DATA_PATH,
            PROCESSED_DATA_PATH, DBSCAN_MODEL_PATH,
        )
        return jsonify({
            "status": "ok",
            "models_loaded": {
                "random_forest": os.path.exists(RF_MODEL_PATH),
                "dbscan_labels": os.path.exists(DBSCAN_MODEL_PATH),
                "ari_data": os.path.exists(ARI_DATA_PATH),
                "processed_data": os.path.exists(PROCESSED_DATA_PATH),
            },
        })

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print(f"\n  Starting Accident Hotspot API (India) on http://localhost:{port}")
    print(f"  Debug mode: {debug}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
