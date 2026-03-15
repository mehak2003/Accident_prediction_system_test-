import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "road.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_accidents.csv")
EDA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "eda")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
DBSCAN_MODEL_PATH = os.path.join(MODELS_DIR, "dbscan_labels.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURE_IMPORTANCES_PATH = os.path.join(MODELS_DIR, "feature_importances.joblib")
CLUSTER_DATA_PATH = os.path.join(MODELS_DIR, "cluster_data.joblib")
ARI_DATA_PATH = os.path.join(MODELS_DIR, "ari_data.joblib")

# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "accident_hotspot_db"),
}

# ---------------------------------------------------------------------------
# DBSCAN hyper-parameters  (tuned for Indian geocoded coordinates)
# ---------------------------------------------------------------------------
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", 0.045))       # ~5 km radius
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 15))

# ---------------------------------------------------------------------------
# Random Forest hyper-parameters
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", 200))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", 20))
RF_TEST_SIZE = float(os.getenv("RF_TEST_SIZE", 0.2))

# ---------------------------------------------------------------------------
# ARI risk-tier thresholds
# ---------------------------------------------------------------------------
ARI_TIERS = {
    "Low": (0.0, 0.3),
    "Moderate": (0.3, 0.5),
    "Severe": (0.5, 0.7),
    "Critical": (0.7, 1.0),
}

# ---------------------------------------------------------------------------
# Severity mapping  (raw labels → numeric)
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    "Slight Injury": 1,
    "Serious Injury": 2,
    "Fatal Injury": 3,
    "Fatal injury": 3,
    "Slight injury": 1,
    "Serious injury": 2,
}
