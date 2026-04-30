"""API routes for CSV dataset upload, pipeline trigger, and upload history."""

import json
import os
import sys
import traceback
import uuid
from datetime import datetime

from flask import Blueprint, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_PATH, MODELS_DIR, EDA_OUTPUT_DIR, PROCESSED_DATA_PATH,
)

upload_bp = Blueprint("upload", __name__)

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "upload_history.json")


# ─────────────────────────────────────────────────────────────────────────────
# History helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(records):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def _update_entry(entry_id, updates):
    records = _load_history()
    for r in records:
        if r["id"] == entry_id:
            r.update(updates)
            break
    _save_history(records)


def _clear_artifacts():
    """Remove all generated model files and EDA JSONs."""
    removed = []
    for path in [PROCESSED_DATA_PATH]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    for folder in [MODELS_DIR, EDA_OUTPUT_DIR]:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    removed.append(fpath)
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@upload_bp.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    f.save(RAW_DATA_PATH)
    size = os.path.getsize(RAW_DATA_PATH)

    # Validate CSV format
    try:
        import pandas as pd
        df = pd.read_csv(RAW_DATA_PATH, nrows=5)
        
        # Required columns for the accident dataset
        required_cols = [
            'Accident_severity', 'Time', 'Day_of_week', 'Area_accident_occured',
            'Weather_conditions', 'Light_conditions', 'Road_surface_conditions',
            'Type_of_collision', 'Cause_of_accident'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            os.remove(RAW_DATA_PATH)  # Remove invalid file
            return jsonify({
                "error": f"Invalid CSV format. Missing required columns: {', '.join(missing_cols)}",
                "hint": "This system requires detailed accident records, not summary statistics. See the Data page for the complete list of 19 required columns.",
                "your_columns": list(df.columns),
                "required_columns": required_cols
            }), 400
            
    except Exception as e:
        os.remove(RAW_DATA_PATH)
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    entry = {
        "id": str(uuid.uuid4()),
        "filename": f.filename,
        "uploaded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "size_bytes": size,
        "status": "uploaded",
        "pipeline_result": None,
    }
    records = _load_history()
    records.insert(0, entry)          # newest first
    _save_history(records)

    return jsonify({
        "message": "File uploaded successfully",
        "id": entry["id"],
        "filename": f.filename,
        "size_bytes": size,
    })


@upload_bp.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """Run the full ML pipeline. Updates the most-recent history entry."""
    # Find the latest upload entry to attach results to
    records = _load_history()
    latest = records[0] if records else None
    entry_id = latest["id"] if latest else None

    if entry_id:
        _update_entry(entry_id, {"status": "running"})

    try:
        from scripts.preprocess import run as preprocess_run
        from scripts.eda import run as eda_run
        from scripts.clustering import run as clustering_run
        from scripts.classifier import run as classifier_run
        from scripts.ari import run as ari_run

        preprocess_run()
        eda_run()
        cluster_summary = clustering_run()
        clf, _ = classifier_run()
        ari_df = ari_run()

        # Collect result stats
        import json as _json
        metrics_path = os.path.join(EDA_OUTPUT_DIR, "model_metrics.json")
        accuracy = None
        if os.path.exists(metrics_path):
            with open(metrics_path) as mf:
                accuracy = _json.load(mf).get("accuracy")

        n_clusters = int(len(ari_df)) if ari_df is not None else None
        ari_min = round(float(ari_df["ARI_Score"].min()), 4) if ari_df is not None else None
        ari_max = round(float(ari_df["ARI_Score"].max()), 4) if ari_df is not None else None

        import pandas as pd
        n_records = None
        if os.path.exists(PROCESSED_DATA_PATH):
            n_records = int(len(pd.read_csv(PROCESSED_DATA_PATH)))

        pipeline_result = {
            "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "records": n_records,
            "clusters": n_clusters,
            "accuracy": accuracy,
            "ari_range": [ari_min, ari_max],
        }

        if entry_id:
            _update_entry(entry_id, {"status": "complete", "pipeline_result": pipeline_result})

        return jsonify({"message": "Pipeline completed successfully", "result": pipeline_result})

    except Exception as exc:
        if entry_id:
            _update_entry(entry_id, {
                "status": "failed",
                "pipeline_result": {"error": str(exc)},
            })
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@upload_bp.route("/api/uploads", methods=["GET"])
def get_uploads():
    """Return full upload history (newest first)."""
    return jsonify(_load_history())


@upload_bp.route("/api/uploads/<entry_id>", methods=["DELETE"])
def delete_upload(entry_id):
    """
    Remove a history entry.
    Pass ?clear_artifacts=true to also wipe all generated model/EDA files.
    """
    records = _load_history()
    match = next((r for r in records if r["id"] == entry_id), None)
    if not match:
        return jsonify({"error": "Entry not found"}), 404

    records = [r for r in records if r["id"] != entry_id]
    _save_history(records)

    cleared = []
    if request.args.get("clear_artifacts", "").lower() == "true":
        cleared = _clear_artifacts()
        # Also remove the raw CSV
        if os.path.exists(RAW_DATA_PATH):
            os.remove(RAW_DATA_PATH)
            cleared.append(RAW_DATA_PATH)

    return jsonify({
        "message": "Entry deleted",
        "id": entry_id,
        "artifacts_cleared": len(cleared),
    })
