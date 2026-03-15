# AI-Based Accident Hotspot Prediction System — Backend (India)

A proactive road-safety intelligence system that integrates **GIS spatial analysis** with a **hybrid Machine Learning pipeline** (DBSCAN + Random Forest) to identify accident Black Spots across India and predict severity under varying environmental and road conditions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dataset](#dataset)
3. [ML Pipeline & Models](#ml-pipeline--models)
4. [Tech Stack](#tech-stack)
5. [Setup & Installation](#setup--installation)
6. [Running the ML Pipeline](#running-the-ml-pipeline)
7. [Starting the API Server](#starting-the-api-server)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Project Structure](#project-structure)

---

## Architecture Overview

```
road.csv (Indian Accident Data)
    │
    ▼
┌──────────────────┐    ┌──────────┐    ┌──────────────────┐
│  Preprocessing    │───▶│   EDA    │    │     DBSCAN       │
│  • Clean & impute │    │  (stats, │    │  (spatial         │
│  • Geocode areas  │    │   JSON)  │    │   clustering on   │
│    → lat/lng      │    └──────────┘    │   geocoded coords)│
│  • Encode cats    │                    └────────┬─────────┘
│  • Engineer feats │                             │
└────────┬─────────┘                             │
         │                                        ▼
         │              ┌────────────────────────────────────┐
         │              │   Random Forest Classifier          │
         │              │   (17 features incl. Cluster_ID)    │
         │              │   → Severity: Slight/Serious/Fatal  │
         │              └────────────────┬───────────────────┘
         │                               │
         ▼                               ▼
┌────────────────────────────────────────────────────┐
│          Accident Risk Index (ARI)                  │
│   W1·Severity + W2·Density + W3·Environment        │
│   Weights derived from RF feature importances       │
│   → Risk Tiers: Low / Moderate / Severe / Critical  │
└────────────────────────┬───────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌───────────┐  ┌──────────────────────┐
    │  MySQL   │  │  .joblib  │  │  Flask REST API       │
    │  (3 tbl) │  │  models   │  │  JSON / GeoJSON       │
    └──────────┘  └───────────┘  │  14+ endpoints        │
                                 └──────────────────────┘
```

---

## Dataset

### Road Accident Severity in India

| Property        | Details                                                      |
|-----------------|--------------------------------------------------------------|
| **Source**      | [Kaggle — Road Accident Severity in India](https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india) |
| **File**        | `road.csv`                                                   |
| **Format**      | CSV                                                          |
| **Target**      | `Accident_severity` (Slight Injury / Serious Injury / Fatal Injury) |

#### Key Columns Used

| Column                        | Type        | Used For                          |
|-------------------------------|-------------|-----------------------------------|
| `Accident_severity`           | Categorical | **Target variable** (mapped to 1/2/3) |
| `Time`                        | String      | Extracted `Hour` feature          |
| `Day_of_week`                 | Categorical | `DayOfWeek` numeric (0–6)        |
| `Area_accident_occured`       | Categorical | **Geocoded to Latitude/Longitude** |
| `Weather_conditions`          | Categorical | Binned to Clear/Rain/Fog/Snow/Wind/Other |
| `Light_conditions`            | Categorical | `Is_Night` binary + encoded feature |
| `Road_surface_type`           | Categorical | Encoded for RF model              |
| `Road_surface_conditions`     | Categorical | Encoded for RF model              |
| `Type_of_collision`           | Categorical | Encoded for RF model              |
| `Type_of_vehicle`             | Categorical | Encoded for RF model              |
| `Cause_of_accident`           | Categorical | Encoded for RF model              |
| `Road_allignment`             | Categorical | Encoded for RF model              |
| `Types_of_Junction`           | Categorical | Encoded for RF model              |
| `Lanes_or_Medians`            | Categorical | Encoded for RF model              |
| `Number_of_vehicles_involved` | Numeric     | Direct feature                    |
| `Driving_experience`          | Categorical | Encoded for RF model              |
| `Age_band_of_driver`          | Categorical | Encoded for RF model              |
| `Vehicle_movement`            | Categorical | Encoded for RF model              |
| `Pedestrian_movement`         | Categorical | Encoded for RF model              |

#### Geocoding Approach

The raw dataset does **not** contain latitude/longitude coordinates. The `Area_accident_occured` column contains area types (e.g., "Office areas", "Residential areas", "Market areas", "Rural village areas"). The preprocessing pipeline maps these to representative Indian city coordinates with random spatial jitter (~2–8 km) to create a realistic geographic distribution suitable for DBSCAN spatial clustering:

| Area Type           | Mapped City   | Approx. Coordinates   |
|---------------------|---------------|-----------------------|
| Office areas        | Delhi NCR     | 28.63°N, 77.22°E     |
| Residential areas   | Mumbai        | 19.08°N, 72.88°E     |
| Church areas        | Bangalore     | 12.93°N, 77.61°E     |
| Industrial areas    | Ahmedabad     | 23.02°N, 72.57°E     |
| School areas        | Chennai       | 13.08°N, 80.27°E     |
| Recreational areas  | Kolkata       | 22.57°N, 88.36°E     |
| Hospital areas      | Jaipur        | 26.91°N, 75.79°E     |
| Market areas        | Hyderabad     | 17.39°N, 78.49°E     |
| Rural village areas | Nagpur        | 21.15°N, 79.09°E     |
| Outside rural areas | Varanasi      | 25.32°N, 82.97°E     |

#### How to Download

1. Create a free [Kaggle](https://www.kaggle.com) account.
2. Visit: https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india
3. Click **Download** and extract the archive.
4. Place `road.csv` at: **`backend/data/road.csv`**

---

## ML Pipeline & Models

### 1. Data Preprocessing (`scripts/preprocess.py`)

- Maps `Accident_severity` labels to numeric (1=Slight, 2=Serious, 3=Fatal)
- Fills missing categorical values with "Unknown"
- **Geocodes** `Area_accident_occured` to (latitude, longitude) with spatial jitter
- Extracts `Hour` from the `Time` column
- Maps `Day_of_week` to numeric 0–6
- Creates `Is_Night` binary from `Light_conditions`
- Bins `Weather_conditions` into 6 categories: Clear, Rain, Fog, Snow, Wind, Other
- Label-encodes 15 categorical features; serialises encoders via joblib

### 2. Exploratory Data Analysis (`scripts/eda.py`)

Generates 11 pre-aggregated JSON files:
- Accident frequency by hour, day of week
- Severity distribution (Slight/Serious/Fatal)
- Weather condition distribution
- Top areas by accident count
- Collision type distribution
- Cause of accident distribution
- Vehicle type distribution
- Severity cross-tabulated with weather and light conditions
- Summary statistics (total records, ranges, etc.)

### 3. DBSCAN Spatial Clustering (`scripts/clustering.py`)

| Parameter       | Default | Rationale                                           |
|-----------------|---------|------------------------------------------------------|
| `eps`           | 0.045 rad | ~5 km radius (calibrated for geocoded Indian areas)|
| `min_samples`   | 15      | Minimum accidents to form a Black Spot              |
| `metric`        | haversine | Geographically accurate distance on lat/lng        |
| `algorithm`     | ball_tree | Efficient for haversine metric                    |

**Why DBSCAN over K-Means?**
- Discovers clusters of arbitrary shape (accidents along curved roads/highways)
- Automatically identifies noise/outliers (isolated random incidents)
- Does not require pre-specifying the number of clusters

**Output:** Cluster centroids (mean lat/lng), incident counts, dominant weather, dominant area per cluster.

### 4. Random Forest Classifier (`scripts/classifier.py`)

| Parameter         | Value       | Rationale                                     |
|-------------------|-------------|------------------------------------------------|
| `n_estimators`    | 200         | Sufficient ensemble diversity                  |
| `max_depth`       | 20          | Captures non-linear feature interactions       |
| `class_weight`    | balanced    | Addresses class imbalance across severity levels |
| `test_size`       | 0.2         | 80/20 stratified train-test split              |

**Feature Matrix (17 features):**

```
Hour, DayOfWeek, Is_Night, Weather_Binned_Enc, Num_Vehicles,
Type_of_vehicle_Enc, Road_surface_type_Enc, Road_surface_conditions_Enc,
Light_conditions_Enc, Type_of_collision_Enc, Cause_of_accident_Enc,
Road_allignment_Enc, Types_of_Junction_Enc, Lanes_or_Medians_Enc,
Driving_experience_Enc, Age_band_of_driver_Enc, Cluster_ID
```

**Key Innovation:** `Cluster_ID` from DBSCAN is included as a feature, making the classifier spatially aware — it learns that severity patterns differ between geographic clusters (urban vs. rural, highway vs. intersection).

**Target:** `Severity` (1=Slight Injury, 2=Serious Injury, 3=Fatal Injury)

**Evaluation:** Accuracy, per-class precision/recall/F1, confusion matrix, and feature importances are all saved to `data/eda/model_metrics.json`.

### 5. Accident Risk Index (`scripts/ari.py`)

```
ARI = W1 × Severity_Score + W2 × Accident_Density + W3 × Environmental_Factor
```

| Component            | Computation                                         |
|----------------------|-----------------------------------------------------|
| Severity Score       | Mean predicted severity per cluster, normalised 0–1 |
| Accident Density     | Incident count / max count across all clusters      |
| Environmental Factor | Risk score based on dominant weather in cluster     |

**Weather Risk Scores:**

| Weather  | Risk Score |
|----------|------------|
| Clear    | 0.15       |
| Wind     | 0.55       |
| Rain     | 0.75       |
| Fog      | 0.80       |
| Snow     | 0.85       |
| Other    | 0.50       |

**Weights (W1, W2, W3)** are derived from Random Forest `feature_importances_` — they adapt to whatever patterns the model finds in the Indian data.

| Risk Tier  | ARI Range    | Map Color |
|------------|--------------|-----------|
| Low        | 0.00 – 0.30  | Green     |
| Moderate   | 0.30 – 0.50  | Yellow    |
| Severe     | 0.50 – 0.70  | Orange    |
| Critical   | 0.70 – 1.00  | Red       |

---

## Tech Stack

| Layer         | Technology                                |
|---------------|-------------------------------------------|
| Language      | Python 3.10+                              |
| Web Framework | Flask 3.x + Flask-CORS                    |
| ML Library    | scikit-learn 1.4+ (DBSCAN, RandomForest)  |
| Data          | pandas, NumPy                             |
| Serialisation | joblib                                    |
| Database      | MySQL 8.x (optional — API falls back to file-based serving) |
| GeoJSON       | `geojson` Python library                  |

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- **MySQL 8.x** (optional — system works without it using `.joblib` artefacts)

### Step-by-step

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Accident_prediction_system_test-

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Download the dataset
#    → Go to https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india
#    → Download and extract
#    → Place road.csv at: backend/data/road.csv

# 5. (Optional) Set up MySQL
cd backend
python -c "from utils.db import init_database; init_database()"
```

### Environment Variables (Optional)

Override defaults from `config.py`:

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=yourpassword
export MYSQL_DATABASE=accident_hotspot_db
export DBSCAN_EPS=0.045         # Cluster radius in radians (~5 km)
export DBSCAN_MIN_SAMPLES=15    # Min points per cluster
export RF_N_ESTIMATORS=200
export RF_MAX_DEPTH=20
export FLASK_PORT=5000
export FLASK_DEBUG=1
```

---

## Running the ML Pipeline

From the `backend/` directory:

```bash
# Full pipeline (preprocessing → EDA → DBSCAN → Random Forest → ARI)
python run_pipeline.py

# Full pipeline + seed MySQL database
python run_pipeline.py --seed-db

# Skip preprocessing (reuse existing processed CSV)
python run_pipeline.py --skip-preprocess
```

### What happens during the pipeline

```
STEP 1/5 — Data Preprocessing
  • Loads road.csv, maps severity labels, geocodes areas
  • Engineers temporal features, encodes 15 categoricals
  • Output: data/processed_accidents.csv + models/label_encoders.joblib

STEP 2/5 — Exploratory Data Analysis
  • Computes 11 frequency distributions
  • Output: data/eda/*.json (11 files)

STEP 3/5 — DBSCAN Spatial Clustering
  • Clusters geocoded coordinates using haversine distance
  • Output: models/dbscan_labels.joblib + models/cluster_data.joblib

STEP 4/5 — Random Forest Classification
  • Trains on 17 features (clustered data only), balanced class weights
  • Output: models/rf_model.joblib + models/feature_importances.joblib
  •         data/eda/model_metrics.json

STEP 5/5 — Accident Risk Index
  • Computes ARI per cluster using RF-derived weights
  • Output: models/ari_data.joblib + data/eda/ari_results.json
```

### Running Individual Steps

```bash
python scripts/preprocess.py
python scripts/eda.py
python scripts/clustering.py
python scripts/classifier.py
python scripts/ari.py
python scripts/seed_db.py       # Requires MySQL
```

---

## Starting the API Server

```bash
cd backend
python app.py
```

Server starts at **http://localhost:5000**. Visit the root URL to see all available endpoints.

---

## API Reference

### Core Endpoints

| Method | Endpoint                     | Description                                  |
|--------|------------------------------|----------------------------------------------|
| GET    | `/`                          | Lists all available API endpoints            |
| GET    | `/api/health`                | Health check — shows which models are loaded |
| GET    | `/api/clusters`              | All clusters as GeoJSON FeatureCollection    |
| GET    | `/api/clusters?format=json`  | All clusters as raw JSON array               |
| GET    | `/api/clusters/<id>`         | Single cluster detail + accident records     |
| POST   | `/api/predict`               | Predict severity for given conditions        |

### EDA Endpoints

| Method | Endpoint                        | Description                        |
|--------|---------------------------------|------------------------------------|
| GET    | `/api/eda/hourly`               | Accident count by hour (0–23)      |
| GET    | `/api/eda/weekly`               | Accident count by day of week      |
| GET    | `/api/eda/severity`             | Severity level distribution        |
| GET    | `/api/eda/weather`              | Accident count by weather type     |
| GET    | `/api/eda/top_areas`            | Top areas by accident count        |
| GET    | `/api/eda/collision_types`      | Collision type distribution        |
| GET    | `/api/eda/causes`               | Cause of accident distribution     |
| GET    | `/api/eda/vehicle_types`        | Vehicle type distribution          |
| GET    | `/api/eda/severity_by_weather`  | Severity x Weather cross-tab       |
| GET    | `/api/eda/severity_by_light`    | Severity x Light conditions        |
| GET    | `/api/eda/summary`              | Dataset summary statistics         |
| GET    | `/api/model/metrics`            | RF accuracy, confusion matrix, feature importances |

### Upload & Pipeline

| Method | Endpoint             | Description                                |
|--------|----------------------|--------------------------------------------|
| POST   | `/api/upload`        | Upload a new CSV dataset (multipart form)  |
| POST   | `/api/pipeline/run`  | Trigger full ML pipeline (blocking)        |

### Prediction Request Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**

```json
{
  "cluster_id": 3,
  "predicted_severity": 2,
  "predicted_label": "Serious Injury",
  "severity_probabilities": {
    "Slight Injury": 0.28,
    "Serious Injury": 0.52,
    "Fatal Injury": 0.20
  },
  "ari_score": 0.5841,
  "risk_tier": "Severe",
  "weights": {
    "W1_severity": 0.42,
    "W2_density": 0.23,
    "W3_environment": 0.35
  },
  "input_conditions": {
    "weather": "Rain",
    "hour": 17,
    "env_score": 0.75
  }
}
```

---

## Configuration

All configuration is centralised in `backend/config.py`:

| Parameter            | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `DBSCAN_EPS`         | `0.045` | Cluster radius in radians (~5 km)       |
| `DBSCAN_MIN_SAMPLES` | `15`    | Min accidents to form a Black Spot      |
| `RF_N_ESTIMATORS`    | `200`   | Number of trees in Random Forest        |
| `RF_MAX_DEPTH`       | `20`    | Maximum depth per decision tree         |
| `RF_TEST_SIZE`       | `0.2`   | Test set fraction (stratified split)    |
| `ARI_TIERS`          | See code| Risk tier thresholds for ARI score      |
| `SEVERITY_MAP`       | See code| Maps text labels to 1/2/3               |

---

## Project Structure

```
backend/
├── app.py                  # Flask application entry point
├── config.py               # Centralised configuration
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Master pipeline runner (all 5 ML steps)
├── data/
│   ├── road.csv            # Raw dataset (download from Kaggle)
│   ├── processed_accidents.csv  # Output of preprocessing
│   └── eda/                # Pre-computed JSON statistics
│       ├── hourly.json
│       ├── weekly.json
│       ├── severity.json
│       ├── weather.json
│       ├── top_areas.json
│       ├── collision_types.json
│       ├── causes.json
│       ├── vehicle_types.json
│       ├── severity_by_weather.json
│       ├── severity_by_light.json
│       ├── summary.json
│       ├── model_metrics.json
│       └── ari_results.json
├── models/                 # Serialised ML artefacts
│   ├── rf_model.joblib
│   ├── dbscan_labels.joblib
│   ├── cluster_data.joblib
│   ├── ari_data.joblib
│   ├── label_encoders.joblib
│   └── feature_importances.joblib
├── scripts/                # Modular ML pipeline scripts
│   ├── preprocess.py       # Step 1: Clean, geocode, encode, engineer
│   ├── eda.py              # Step 2: Exploratory Data Analysis
│   ├── clustering.py       # Step 3: DBSCAN spatial clustering
│   ├── classifier.py       # Step 4: Random Forest training
│   ├── ari.py              # Step 5: Accident Risk Index
│   └── seed_db.py          # Step 6: MySQL database seeder
├── routes/                 # Flask API blueprints
│   ├── clusters.py         # /api/clusters endpoints
│   ├── predictions.py      # /api/predict endpoint
│   ├── eda_routes.py       # /api/eda/* endpoints
│   └── upload.py           # /api/upload + /api/pipeline/run
└── utils/
    ├── db.py               # MySQL connection & schema bootstrap
    └── geojson_utils.py    # Cluster/accident → GeoJSON conversion
```

---

## MySQL Database Schema (Optional)

Three tables following the SRS specification:

**`tbl_Spatial_Clusters`** — DBSCAN-identified Black Spots
- `Cluster_ID` (PK), `Centroid_Lat`, `Centroid_Lon`, `Radius_Eps`, `Incident_Count`

**`tbl_Accident_Records`** — Individual accidents linked to clusters
- `Record_ID` (PK), `Latitude`, `Longitude`, `Timestamp`, `Weather_Cond`, `Severity_Hist`, `Cluster_ID` (FK)

**`tbl_Risk_Assessments`** — ARI scores and risk tiers per cluster
- `Assessment_ID` (PK), `Cluster_ID` (FK), `Pred_Severity`, `ARI_Score`, `Risk_Tier`, `Env_Modifier`

> MySQL is **optional**. The API automatically falls back to serving from `.joblib` files when the database is unavailable.

---

## Academic References

1. Ester, M., et al. "A density-based algorithm for discovering clusters in large spatial databases with noise." KDD (1996).
2. Breiman, L. "Random Forests." Machine Learning 45.1 (2001): 5–32.
3. IEEE 830 / ISO/IEC/IEEE 29148 — Software Requirements Specification standards.
