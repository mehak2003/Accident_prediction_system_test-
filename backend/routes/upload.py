"""API route for CSV dataset upload and pipeline trigger."""

from flask import Blueprint, jsonify, request
import os, sys, shutil, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DATA_PATH

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/api/upload", methods=["POST"])
def upload_csv():
    """
    Upload a CSV dataset.  After upload, the user should trigger the
    pipeline via /api/pipeline/run (or run it manually from the CLI).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    f.save(RAW_DATA_PATH)

    return jsonify({
        "message": "File uploaded successfully",
        "path": RAW_DATA_PATH,
        "size_bytes": os.path.getsize(RAW_DATA_PATH),
    })


@upload_bp.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """Run the full ML pipeline (preprocess → EDA → cluster → classify → ARI).
    This is a blocking call and may take several minutes on large datasets."""
    try:
        from scripts.preprocess import run as preprocess_run
        from scripts.eda import run as eda_run
        from scripts.clustering import run as clustering_run
        from scripts.classifier import run as classifier_run
        from scripts.ari import run as ari_run

        preprocess_run()
        eda_run()
        clustering_run()
        classifier_run()
        ari_run()

        return jsonify({"message": "Pipeline completed successfully"})
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }), 500
