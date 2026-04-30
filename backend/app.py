"""
Flask Application — AI-Based Accident Hotspot Prediction System (India)
========================================================================
Entry point for the REST API server.  Registers all route blueprints
and configures CORS for frontend consumption.
"""

import os, sys
from flask import Flask, jsonify
from flask_cors import CORS
from flask_compress import Compress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.clusters import clusters_bp
from routes.predictions import predictions_bp
from routes.eda_routes import eda_bp
from routes.upload import upload_bp


def create_app():
    app = Flask(__name__)

    # Enable gzip compression for all responses
    Compress(app)

    # Production configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.config['COMPRESS_MIMETYPES'] = [
        'text/html', 'text/css', 'text/xml', 'application/json',
        'application/javascript', 'text/javascript'
    ]
    app.config['COMPRESS_LEVEL'] = 6
    app.config['COMPRESS_MIN_SIZE'] = 500

    # Configure CORS for production and development
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", ""),  # Production frontend URL
    ]

    # Remove empty strings from allowed origins
    allowed_origins = [origin for origin in allowed_origins if origin]

    CORS(app,
         origins=allowed_origins,
         supports_credentials=True,
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )

    app.register_blueprint(clusters_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(upload_bp)

    @app.route("/")
    def index():
        return jsonify({
            "service": "Accident Hotspot Prediction API",
            "version": "1.0.0",
            "status": "ok"
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
                "rf": os.path.exists(RF_MODEL_PATH),
                "dbscan": os.path.exists(DBSCAN_MODEL_PATH),
                "ari": os.path.exists(ARI_DATA_PATH),
                "data": os.path.exists(PROCESSED_DATA_PATH),
            },
        })

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print(f"\n  Starting API on http://localhost:{port}")
    print(f"  Debug: {debug}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
