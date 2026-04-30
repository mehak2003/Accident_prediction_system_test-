# AcciHotspot — Complete System Documentation

> **Audience**: New contributors, reviewers, or anyone trying to understand the full system from scratch.  
> **Purpose**: End-to-end walkthrough of every flow, technology, data transformation, and file in the repository.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Technology Stack](#3-technology-stack)
4. [High-Level Architecture Diagram](#4-high-level-architecture-diagram)
5. [End-to-End User Journey](#5-end-to-end-user-journey)
6. [Flow 1 — Data Ingestion & Upload](#6-flow-1--data-ingestion--upload)
7. [Flow 2 — ML Pipeline (5 Stages)](#7-flow-2--ml-pipeline-5-stages)
   - [Stage 1: Preprocessing & Data Cleaning](#stage-1-preprocessing--data-cleaning)
   - [Stage 2: Exploratory Data Analysis (EDA)](#stage-2-exploratory-data-analysis-eda)
   - [Stage 3: DBSCAN Spatial Clustering](#stage-3-dbscan-spatial-clustering)
   - [Stage 4: Random Forest Classifier](#stage-4-random-forest-classifier)
   - [Stage 5: Accident Risk Index (ARI) Scoring](#stage-5-accident-risk-index-ari-scoring)
8. [Flow 3 — Prediction Request](#8-flow-3--prediction-request)
9. [Flow 4 — Frontend Visualization Flows](#9-flow-4--frontend-visualization-flows)
10. [Data Cleaning — In-Depth](#10-data-cleaning--in-depth)
11. [Database & Storage Schema](#11-database--storage-schema)
12. [API Reference](#12-api-reference)
13. [File-by-File Reference](#13-file-by-file-reference)
14. [Configuration Reference](#14-configuration-reference)
15. [Local Setup Guide](#15-local-setup-guide)

---

## 1. Project Overview

**AcciHotspot** is a full-stack road accident analytics and prediction platform. It ingests road accident CSV data, runs a 5-stage machine learning pipeline, and serves an interactive web dashboard that shows:

- Geographic **"Black Spot" clusters** (areas with high accident concentration) on a Leaflet map
- **Severity predictions** for any input conditions via a trained Random Forest model
- **Accident Risk Index (ARI)** scores that rank clusters from Low → Critical
- **Analytics charts** for temporal, environmental, and collision patterns

The platform is designed around a real-world problem: transport authorities or urban planners upload raw accident data, the system identifies high-risk zones automatically, and any field officer can query predicted severity for specific conditions without writing any code.

---

## 2. Repository Structure

```
Accident_prediction_system_test-/
│
├── README.md                         # Quick-start & API docs
├── DOCUMENTATION.md                  # ← This file
│
├── backend/                          # Python Flask REST API + ML pipeline
│   ├── app.py                        # Server entry point (port 5000)
│   ├── config.py                     # All tunable constants (paths, hyperparameters)
│   ├── run_pipeline.py               # CLI runner for the ML pipeline
│   ├── research_evaluation.py        # Script generating IEEE paper figures/metrics
│   │
│   ├── scripts/                      # ML pipeline stages (run in sequence)
│   │   ├── preprocess.py             # Stage 1: Clean, geocode, encode raw CSV
│   │   ├── eda.py                    # Stage 2: Aggregate stats → JSON files
│   │   ├── clustering.py             # Stage 3: DBSCAN spatial clustering
│   │   ├── classifier.py             # Stage 4: Random Forest severity model
│   │   ├── ari.py                    # Stage 5: Accident Risk Index scoring
│   │   └── seed_db.py                # Stage 6 (optional): Seed MySQL from joblib
│   │
│   ├── routes/                       # Flask route handlers (blueprints)
│   │   ├── clusters.py               # GET /api/clusters, GET /api/clusters/<id>
│   │   ├── predictions.py            # POST /api/predict
│   │   ├── eda.py                    # GET /api/eda/*
│   │   ├── uploads.py                # POST /api/upload, GET/DELETE /api/uploads
│   │   └── pipeline.py               # POST /api/pipeline/run
│   │
│   ├── utils/
│   │   ├── geojson_utils.py          # Converts cluster DataFrames → GeoJSON
│   │   └── db.py                     # MySQL connection + schema bootstrap
│   │
│   ├── data/
│   │   ├── road.csv                  # Raw upload (replaced on each upload)
│   │   ├── processed_accidents.csv   # Output of preprocess stage
│   │   ├── upload_history.json       # Persistent upload metadata log
│   │   └── eda/                      # JSON outputs from EDA stage (13 files)
│   │       ├── hourly.json
│   │       ├── weekly.json
│   │       ├── severity.json
│   │       ├── weather.json
│   │       ├── top_areas.json
│   │       ├── collision_types.json
│   │       ├── causes.json
│   │       ├── vehicle_types.json
│   │       ├── severity_by_weather.json
│   │       ├── severity_by_light.json
│   │       ├── summary.json
│   │       ├── model_metrics.json    # RF accuracy, confusion matrix
│   │       └── ari_results.json      # ARI scores per cluster
│   │
│   ├── models/                       # Serialized ML artifacts (joblib)
│   │   ├── label_encoders.joblib     # LabelEncoder instances for 14 columns
│   │   ├── dbscan_labels.joblib      # Per-row cluster assignments
│   │   ├── cluster_data.joblib       # Cluster summary DataFrame
│   │   ├── rf_model.joblib           # Trained Random Forest classifier
│   │   ├── feature_importances.joblib # Feature importance dict
│   │   └── ari_data.joblib           # Cluster DataFrame + ARI scores
│   │
│   └── requirements.txt              # Python dependencies
│
├── frontend/                         # React + Vite SPA (port 5173)
│   ├── index.html                    # HTML shell with <div id="root">
│   ├── vite.config.js                # Vite config + /api proxy to :5000
│   ├── package.json                  # Node dependencies & scripts
│   │
│   ├── src/
│   │   ├── main.jsx                  # React DOM render entry
│   │   ├── App.jsx                   # Router + sidebar layout + theme toggle
│   │   ├── api.js                    # Axios instance + all API call functions
│   │   ├── ThemeContext.jsx           # Dark/light mode context + localStorage
│   │   ├── index.css                 # Global CSS variables & resets
│   │   │
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx         # Route "/" — KPI cards + summary charts
│   │   │   ├── MapView.jsx           # Route "/map" — Leaflet interactive map
│   │   │   ├── Analytics.jsx         # Route "/analytics" — EDA chart gallery
│   │   │   ├── Predict.jsx           # Route "/predict" — Severity predictor form
│   │   │   └── DataManager.jsx       # Route "/data" — Upload + pipeline trigger
│   │   │
│   │   └── components/
│   │       ├── StatCard.jsx          # Metric tile (icon, label, value, subtitle)
│   │       ├── ChartCard.jsx         # Wrapper container for Recharts charts
│   │       ├── RiskBadge.jsx         # Colored badge: Low/Moderate/Severe/Critical
│   │       └── Loader.jsx            # Animated loading spinner
│   │
│   └── public/                       # Static assets
│
└── paper/                            # IEEE research paper artifacts
    └── *.pdf / *.tex                 # Paper drafts and generated figures
```

---

## 3. Technology Stack

### Backend

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | All backend logic |
| Web Framework | Flask | 3.0+ | REST API server |
| CORS | Flask-CORS | 4.0+ | Allow cross-origin requests from React |
| Data Processing | pandas | 2.1+ | DataFrame operations for all ML stages |
| Numerics | NumPy | 1.26+ | Array math, radian conversion |
| ML | scikit-learn | 1.4+ | DBSCAN + Random Forest |
| Serialization | joblib | 1.3+ | Save/load ML models and DataFrames |
| Geospatial | GeoJSON | 3.1+ | Format cluster data for map consumption |
| Database | MySQL 8.3+ | optional | Relational storage (file-based by default) |
| MySQL Driver | mysql-connector-python | 8.3+ | Python ↔ MySQL bridge |
| Visualization | matplotlib + seaborn | 3.8+, 0.13+ | Research paper figure generation |

### Frontend

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | JavaScript (ESM) | ES2022 | All frontend logic |
| UI Framework | React | 19.2.4 | Component-based SPA |
| Routing | React Router | 7.13.2 | Client-side routing (5 pages) |
| Build Tool | Vite | 8.0+ | Dev server, hot reload, bundling |
| Styling | Tailwind CSS | 4.2.2 | Utility-first CSS |
| Charts | Recharts | 3.8.1 | Bar, Line, Pie, Radar charts |
| Maps | Leaflet + react-leaflet | 1.9.4 / 5.0.0 | Interactive GIS map |
| HTTP Client | Axios | 1.14.0 | REST calls to backend |
| Icons | react-icons | 5.6.0 | SVG icon library |

---

## 4. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BROWSER (Port 5173)                              │
│                                                                             │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌─────────┐  ┌─────────────┐  │
│  │Dashboard │  │ MapView │  │ Analytics │  │ Predict │  │ DataManager │  │
│  └────┬─────┘  └────┬────┘  └─────┬─────┘  └────┬────┘  └──────┬──────┘  │
│       │              │             │              │              │          │
│       └──────────────┴─────────────┴──────────────┴──────────────┘          │
│                                    │                                         │
│                              api.js (Axios)                                  │
│                              /api/* → proxy                                  │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │ HTTP/JSON
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLASK API SERVER (Port 5000)                        │
│                                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────────────┐ │
│  │clusters.py  │  │predictions.py│  │  eda.py    │  │uploads/pipeline   │ │
│  │GET /clusters│  │POST /predict │  │GET /eda/*  │  │POST /upload       │ │
│  │GET /cluster/│  │              │  │            │  │POST /pipeline/run │ │
│  │    <id>     │  │              │  │            │  │GET/DEL /uploads   │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  └────────┬──────────┘ │
│         │                │                 │                   │            │
│         └────────────────┴─────────────────┴───────────────────┘            │
│                                    │                                         │
│                          MODEL STORE (joblib files)                          │
│   rf_model · label_encoders · cluster_data · ari_data · feature_importances │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │ triggered by POST /pipeline/run
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ML PIPELINE (Sequential)                           │
│                                                                             │
│   [Stage 1]           [Stage 2]           [Stage 3]                        │
│  preprocess.py  ───►   eda.py     ───►  clustering.py                      │
│  road.csv             stats JSON          DBSCAN                            │
│  → processed.csv      (11 files)          → cluster_data.joblib             │
│                                                │                            │
│                                                ▼                            │
│                               [Stage 4]               [Stage 5]            │
│                             classifier.py   ───►      ari.py               │
│                             Random Forest              ARI Score            │
│                             → rf_model.joblib          → ari_data.joblib   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
            FILE STORAGE                      MYSQL (optional)
         backend/data/*.csv              accident_hotspot_db
         backend/data/eda/*.json         tbl_Spatial_Clusters
         backend/models/*.joblib         tbl_Accident_Records
                                         tbl_Risk_Assessments
```

---

## 5. End-to-End User Journey

Below is the complete lifecycle of a user session from first visit to getting a prediction:

```
User opens browser at http://localhost:5173
        │
        ▼
[1] DataManager page (/data)
    User selects a CSV file  ──► POST /api/upload
                                  │ saves road.csv
                                  │ appends to upload_history.json
                                  └─► { id, filename, size_bytes }

        │ User clicks "Run Pipeline"
        ▼
[2] POST /api/pipeline/run  ──► runs all 5 ML stages (30-90 seconds)
    Response: { records: 12316, clusters: 12, accuracy: 84.54%, ari_range: [...] }

        │ User navigates to Dashboard (/)
        ▼
[3] Dashboard fetches:
    GET /api/clusters         ──► 12 GeoJSON features
    GET /api/eda/summary      ──► total records, severity breakdown
    Renders: 4 KPI tiles, bar chart (top clusters by ARI), pie (tier distribution), table

        │ User clicks "Map"
        ▼
[4] MapView (/map) fetches GET /api/clusters
    Renders circle markers on India map; radius ∝ incident count; color = risk tier
    User filters by "Critical" → 3 clusters remain visible
    User clicks cluster "0" popup → shows ARI=0.70, 2060 incidents, Residential/Mumbai

        │ User clicks "Analytics"
        ▼
[5] Analytics (/analytics) fetches 10 EDA endpoints in parallel
    Renders: hourly line chart, weekly bar, severity pie, weather bar, cross-tabs, etc.

        │ User clicks "Predict"
        ▼
[6] Predict (/predict)
    User fills: Cluster=0, Hour=18, Night=1, Weather=Rain, ...
    Submits ──► POST /api/predict
    Response: { predicted_severity: 2 (Serious), probabilities: {1:0.14,2:0.61,3:0.25},
                ari_score: 0.70, risk_tier: "Critical", weights: {...} }
    UI shows colored result card with probability breakdown
```

---

## 6. Flow 1 — Data Ingestion & Upload

### Diagram

```
Browser (DataManager.jsx)
│
│  User picks file → <input type="file" accept=".csv">
│
│  handleUpload():
│    const fd = new FormData()
│    fd.append("file", selectedFile)
│    POST /api/upload  (multipart/form-data)
│
└──────────────────────────────────────────────────────────►
                                                           Flask (routes/uploads.py)
                                                           upload_csv()
                                                             │
                                                             ├─ Validate: file present, .csv extension
                                                             ├─ Save: file.save("backend/data/road.csv")
                                                             ├─ Record: append entry to upload_history.json
                                                             │    { id, filename, size_bytes,
                                                             │      uploaded_at, status:"uploaded" }
                                                             └─► 200 { id, filename, size_bytes }
◄──────────────────────────────────────────────────────────
│  UI shows green badge "Uploaded successfully"
│  Upload history table refreshes (GET /api/uploads)
```

### What is stored

`upload_history.json` is a flat JSON array, newest first. Each entry:

```json
{
  "id": "abc123",
  "filename": "road_accidents_2023.csv",
  "size_bytes": 2048740,
  "uploaded_at": "2024-01-15T10:23:45",
  "status": "uploaded",
  "completed_at": null,
  "records": null,
  "clusters": null,
  "accuracy": null,
  "ari_range": null
}
```

After the pipeline runs, `status` becomes `"completed"` and the numeric fields are populated.

### Delete behavior

- `DELETE /api/uploads/<id>` — removes only the history entry  
- `DELETE /api/uploads/<id>?clear_artifacts=true` — also deletes `road.csv`, `processed_accidents.csv`, all `eda/*.json` files, and all `models/*.joblib` files

---

## 7. Flow 2 — ML Pipeline (5 Stages)

### Pipeline Orchestration Diagram

```
POST /api/pipeline/run
        │
        ▼
 routes/pipeline.py::run_pipeline()
        │
        ├─[1]──► preprocess.run()   → backend/data/processed_accidents.csv
        │                              backend/models/label_encoders.joblib
        │
        ├─[2]──► eda.run()          → backend/data/eda/[11 json files]
        │
        ├─[3]──► clustering.run()   → backend/models/cluster_data.joblib
        │                              backend/models/dbscan_labels.joblib
        │                              (updates processed_accidents.csv + Cluster_ID column)
        │
        ├─[4]──► classifier.run()   → backend/models/rf_model.joblib
        │                              backend/models/feature_importances.joblib
        │                              backend/data/eda/model_metrics.json
        │
        ├─[5]──► ari.run()          → backend/models/ari_data.joblib
        │                              backend/data/eda/ari_results.json
        │
        └─► update upload_history.json (status=completed, metrics populated)
            return { records, clusters, accuracy, ari_range }
```

---

### Stage 1: Preprocessing & Data Cleaning

**File**: `backend/scripts/preprocess.py`  
**Input**: `backend/data/road.csv` (~12,316 rows, ~30 columns)  
**Output**: `backend/data/processed_accidents.csv` + `backend/models/label_encoders.joblib`

#### What the raw CSV looks like

```
Time,Day_of_week,Age_band_of_driver,Sex_of_driver,Educational_level,
Vehicle_driver_relation,Driving_experience,Lanes_or_Medians,
Types_of_Junction,Road_surface_type,Road_surface_conditions,
Light_conditions,Weather_conditions,Type_of_collision,
Number_of_vehicles_involved,Number_of_casualties,
Area_accident_occured,Lanes_or_Medians,Types_of_Junction,
Vehicle_movement,Casualty_class,Sex_of_casualty,
Age_band_of_casualty,Casualty_severity,Work_of_casuality,
Fitness_of_casuality,Pedestrian_movement,Cause_of_accident,
Type_of_vehicle,Accident_severity
```

#### Full Preprocessing Flow Diagram

```
road.csv (raw)
    │
    ├─[Step 1] Load into DataFrame
    │
    ├─[Step 2] Strip whitespace
    │            df.columns = df.columns.str.strip()
    │            for each object column: col.str.strip()
    │
    ├─[Step 3] Map severity labels to integers
    │            "Slight Injury"  → 1
    │            "Serious Injury" → 2
    │            "Fatal Injury"   → 3
    │            (drop rows where severity is NaN)
    │
    ├─[Step 4] Fill missing values
    │            categorical columns → "Unknown"
    │            Number_of_vehicles_involved → median value
    │
    ├─[Step 5] Parse Time column → Hour (int 0–23)
    │            Try formats in order:
    │              a) "HH:MM:SS" via regex (\d{1,2}):(\d{2})
    │              b) "HH.MM.SS" via regex (\d{1,2})\.(\d{2})
    │              c) decimal hour (e.g. "18.5" → hour=18)
    │              d) fallback → Hour=0
    │            Hour = parsed_hour % 24
    │
    ├─[Step 6] Extract DayOfWeek integer
    │            Monday=0, Tuesday=1, ..., Sunday=6
    │            (from Day_of_week string column)
    │
    ├─[Step 7] Create Is_Night binary feature
    │            Is_Night = 1 if Light_conditions contains "dark" or "night" (case-insensitive)
    │            Is_Night = 0 otherwise
    │
    ├─[Step 8] Bin Weather_conditions → Weather_Binned (6 categories)
    │            "Clear"  ← matches: "Normal", "Clear"
    │            "Rain"   ← matches: "Rain", "Rainy"
    │            "Fog"    ← matches: "Fog", "Mist"
    │            "Snow"   ← matches: "Snow"
    │            "Wind"   ← matches: "Wind"
    │            "Other"  ← everything else
    │
    ├─[Step 9] Geocode Area_accident_occured → Latitude, Longitude
    │            Uses AREA_COORDS lookup table (12 mappings):
    │
    │   Area string          City              Lat        Lon
    │   ─────────────────────────────────────────────────────
    │   "Office areas"    → Delhi NCR       28.6282   77.2195
    │   "Residential"     → Mumbai          19.0760   72.8777
    │   "Church areas"    → Bangalore       12.9352   77.6245
    │   "Industrial"      → Ahmedabad       23.0225   72.5714
    │   "School areas"    → Chennai         13.0827   80.2707
    │   "Recreational"    → Hyderabad       17.3850   78.4867
    │   "Hospital areas"  → Pune            18.5204   73.8567
    │   "Market areas"    → Kolkata         22.5726   88.3639
    │   "Rural"           → Patna           25.5941   85.1376
    │   "Outside city"    → Jaipur          26.9124   75.7873
    │   "Unknown"         → Centre India    20.5937   78.9629
    │   (default)         → Centre India    20.5937   78.9629
    │
    │   Per-row jitter added AFTER geocoding:
    │     seed = hash(area_string) % 1000
    │     np.random.seed(seed)
    │     jitter = np.random.uniform(-0.015, 0.015, size=2)  # ~±1.67 km
    │     Latitude  += jitter[0]
    │     Longitude += jitter[1]
    │   (Ensures nearby accidents for same area stay clusterable by DBSCAN)
    │
    ├─[Step 10] Rename: Number_of_vehicles_involved → Num_Vehicles (int)
    │
    ├─[Step 11] Label-encode 14 categorical columns
    │            Each column gets its own sklearn LabelEncoder:
    │
    │   Column                      Encoded as
    │   ────────────────────────────────────────────────────────
    │   Weather_Binned              Weather_Binned_Enc
    │   Type_of_vehicle             Type_of_vehicle_Enc
    │   Road_surface_type           Road_surface_type_Enc
    │   Road_surface_conditions     Road_surface_conditions_Enc
    │   Light_conditions            Light_conditions_Enc
    │   Type_of_collision           Type_of_collision_Enc
    │   Cause_of_accident           Cause_of_accident_Enc
    │   Area_accident_occured       Area_accident_occured_Enc
    │   Road_allignment             Road_allignment_Enc
    │   Types_of_Junction           Types_of_Junction_Enc
    │   Lanes_or_Medians            Lanes_or_Medians_Enc
    │   Driving_experience          Driving_experience_Enc
    │   Age_band_of_driver          Age_band_of_driver_Enc
    │   Vehicle_movement            Vehicle_movement_Enc
    │
    │   All LabelEncoders saved to: backend/models/label_encoders.joblib
    │   (Required for prediction-time encoding of user inputs)
    │
    └─[Output] processed_accidents.csv (39 columns)
```

#### Output column set (39 columns)

```
Accident_severity (int: 1/2/3)
Latitude, Longitude (float: geocoded + jitter)
Hour (int: 0-23)
DayOfWeek (int: 0-6)
Is_Night (int: 0/1)
Num_Vehicles (int)
Weather_Binned (str: Clear/Rain/Fog/Snow/Wind/Other)
Weather_Binned_Enc (int)
Type_of_vehicle (str) + Type_of_vehicle_Enc (int)
Road_surface_type (str) + Road_surface_type_Enc (int)
Road_surface_conditions (str) + Road_surface_conditions_Enc (int)
Light_conditions (str) + Light_conditions_Enc (int)
Type_of_collision (str) + Type_of_collision_Enc (int)
Cause_of_accident (str) + Cause_of_accident_Enc (int)
Area_accident_occured (str) + Area_accident_occured_Enc (int)
Road_allignment (str) + Road_allignment_Enc (int)
Types_of_Junction (str) + Types_of_Junction_Enc (int)
Lanes_or_Medians (str) + Lanes_or_Medians_Enc (int)
Driving_experience (str) + Driving_experience_Enc (int)
Age_band_of_driver (str) + Age_band_of_driver_Enc (int)
Vehicle_movement (str) + Vehicle_movement_Enc (int)
```

---

### Stage 2: Exploratory Data Analysis (EDA)

**File**: `backend/scripts/eda.py`  
**Input**: `backend/data/processed_accidents.csv`  
**Output**: 11 JSON files in `backend/data/eda/`

```
processed_accidents.csv
        │
        ├─ Group by Hour (0–23)               → eda/hourly.json
        │    [{"hour":0,"count":312}, ...]
        │
        ├─ Group by DayOfWeek (0–6)           → eda/weekly.json
        │    [{"day":"Monday","count":1820}, ...]
        │
        ├─ Group by Accident_severity          → eda/severity.json
        │    [{"label":"Slight","count":7800}, ...]
        │
        ├─ Group by Weather_Binned             → eda/weather.json
        │    [{"weather":"Clear","count":6500}, ...]
        │
        ├─ Group by Area_accident_occured      → eda/top_areas.json
        │    Top 20 areas by count
        │
        ├─ Group by Type_of_collision          → eda/collision_types.json
        │
        ├─ Group by Cause_of_accident          → eda/causes.json
        │
        ├─ Group by Type_of_vehicle            → eda/vehicle_types.json
        │
        ├─ Pivot: Severity × Weather_Binned    → eda/severity_by_weather.json
        │    [{"weather":"Clear","Slight":..., "Serious":..., "Fatal":...}, ...]
        │
        ├─ Pivot: Severity × Light_conditions  → eda/severity_by_light.json
        │
        └─ Summary stats                       → eda/summary.json
             { total_records, severity_distribution,
               lat_min, lat_max, lon_min, lon_max, unique_areas }
```

---

### Stage 3: DBSCAN Spatial Clustering

**File**: `backend/scripts/clustering.py`  
**Input**: `backend/data/processed_accidents.csv`  
**Output**: `backend/models/cluster_data.joblib`, `backend/models/dbscan_labels.joblib`  
**Also**: Adds `Cluster_ID` column to `processed_accidents.csv`

#### What is DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are close together while marking isolated points as noise. Unlike K-Means, it does not require a pre-set number of clusters, and it handles noise naturally.

```
Key parameters:
  eps         = maximum distance between two points to be considered neighbors
  min_samples = minimum points in a neighborhood to form a core point (cluster seed)
```

#### How clustering is applied here

```
processed_accidents.csv (12,316 rows)
        │
        ├─[1] Extract coordinates
        │      coords = df[["Latitude","Longitude"]].values
        │      # shape: (12316, 2)
        │
        ├─[2] Convert degrees → radians
        │      coords_rad = np.radians(coords)
        │      # Required by haversine metric
        │
        ├─[3] Run DBSCAN
        │      DBSCAN(
        │        eps        = 0.005,          # radians ≈ 0.005 × 6371 km = 31.9 km
        │        min_samples = 15,             # at least 15 accidents to form a cluster
        │        algorithm   = "ball_tree",    # efficient spatial index for haversine
        │        metric      = "haversine"     # great-circle distance on Earth's surface
        │      ).fit(coords_rad)
        │
        ├─[4] Result: labels array (one integer per accident row)
        │        -1 = noise (not part of any cluster)
        │        0, 1, 2, ... = cluster ID
        │      Typical result: 12 clusters, ~1,234 noise points
        │
        ├─[5] Filter noise points (label == -1)
        │      clustered_df = df[labels != -1]
        │
        ├─[6] Compute cluster summaries (group by Cluster_ID)
        │      For each cluster:
        │        Centroid_Lat   = mean(Latitude)
        │        Centroid_Lon   = mean(Longitude)
        │        Incident_Count = count(rows)
        │        Mean_Severity  = mean(Accident_severity)
        │        Dominant_Weather = most_frequent(Weather_Binned)
        │        Dominant_Area    = most_frequent(Area_accident_occured)
        │        Radius_Eps     = 0.005 (same for all clusters)
        │
        └─[7] Save outputs
               dbscan_labels.joblib   → raw labels array (12316,)
               cluster_data.joblib    → cluster summary DataFrame (12 rows × 7 cols)
               (also writes Cluster_ID column back to processed_accidents.csv)
```

#### Cluster summary example

```
Cluster_ID  Centroid_Lat  Centroid_Lon  Incident_Count  Mean_Severity  Dominant_Weather  Dominant_Area
0           19.082         72.881         2060            1.78           Clear             Residential
1           28.631         77.223         1840            2.10           Clear             Office areas
2           12.939         77.628         1200            1.65           Rain              Church areas
...         ...            ...            ...             ...            ...               ...
```

---

### Stage 4: Random Forest Classifier

**File**: `backend/scripts/classifier.py`  
**Input**: `backend/data/processed_accidents.csv` (with Cluster_ID)  
**Output**: `backend/models/rf_model.joblib`, `backend/models/feature_importances.joblib`, `backend/data/eda/model_metrics.json`

#### Feature set (17 features)

```
Feature                        Type    Description
──────────────────────────────────────────────────────────────────────────────
Hour                           int     Hour of day (0–23)
DayOfWeek                      int     Day (0=Mon, 6=Sun)
Is_Night                       int     1 if dark/night conditions
Weather_Binned_Enc             int     Encoded weather category
Light_conditions_Enc           int     Encoded light condition
Road_surface_conditions_Enc    int     Encoded road surface state
Road_surface_type_Enc          int     Encoded road surface material
Road_allignment_Enc            int     Encoded road geometry
Types_of_Junction_Enc          int     Encoded junction type
Lanes_or_Medians_Enc           int     Encoded lane/median type
Type_of_collision_Enc          int     Encoded collision type
Num_Vehicles                   int     Number of vehicles involved
Type_of_vehicle_Enc            int     Encoded vehicle type
Cause_of_accident_Enc          int     Encoded cause
Driving_experience_Enc         int     Encoded driver experience band
Age_band_of_driver_Enc         int     Encoded driver age group
Cluster_ID                     int     DBSCAN spatial cluster assignment
```

#### Training flow

```
processed_accidents.csv (with Cluster_ID)
        │
        ├─ X = df[17 feature columns]
        ├─ y = df["Accident_severity"]  (values: 1, 2, 3)
        │
        ├─ train_test_split(X, y,
        │    test_size    = 0.20,
        │    stratify     = y,          # preserves class distribution in both sets
        │    random_state = 42)
        │
        │   → X_train (9,852 rows), X_test (2,464 rows)
        │   → y_train, y_test
        │
        ├─ RandomForestClassifier(
        │    n_estimators  = 200,        # 200 decision trees
        │    max_depth     = 20,         # max tree depth to limit overfitting
        │    class_weight  = "balanced", # auto-adjusts weights ∝ 1/class_freq
        │    random_state  = 42          # reproducibility
        │  ).fit(X_train, y_train)
        │
        ├─ Evaluate on X_test:
        │    accuracy        → ~84.54%
        │    confusion_matrix → 3×3 matrix
        │    classification_report → precision/recall/F1 per class
        │
        ├─ Extract feature importances:
        │    clf.feature_importances_ → [0.18, 0.14, 0.06, ...]
        │    zip(feature_names, importances) → dict
        │    Top 5:
        │      1. Cluster_ID            (0.18)  — spatial location matters most
        │      2. Hour                  (0.14)  — time of day
        │      3. Cause_of_accident_Enc (0.12)  — human error/distraction
        │      4. Weather_Binned_Enc    (0.09)  — environment
        │      5. Road_surface_cond_Enc (0.07)  — infrastructure
        │
        └─ Save:
             rf_model.joblib            → trained RandomForestClassifier
             feature_importances.joblib → { feature: importance } dict
             eda/model_metrics.json     → { accuracy, confusion_matrix,
                                            classification_report,
                                            feature_importances }
```

---

### Stage 5: Accident Risk Index (ARI) Scoring

**File**: `backend/scripts/ari.py`  
**Input**: `backend/models/cluster_data.joblib` + `backend/models/feature_importances.joblib`  
**Output**: `backend/models/ari_data.joblib` + `backend/data/eda/ari_results.json`

#### ARI Formula

```
ARI = W1 × Severity_Score  +  W2 × Density_Score  +  W3 × Env_Score

Where:
  Severity_Score = (Mean_Severity - min_severity) / (max_severity - min_severity)
  Density_Score  = Incident_Count / max_incident_count
  Env_Score      = lookup(Dominant_Weather):
                     Clear  → 0.15
                     Rain   → 0.55
                     Fog    → 0.75
                     Snow   → 0.85
                     Wind   → 0.40
                     Other  → 0.30

  W1, W2, W3 are derived from feature importances:
    W1 (severity weight) ≈ 0.54   ← combined importance of severity-related features
    W2 (density weight)  ≈ 0.39   ← importance of Cluster_ID + spatial features
    W3 (env weight)      ≈ 0.07   ← importance of Weather_Binned_Enc + Light_conditions_Enc

  W1 + W2 + W3 = 1.0  (normalized)
```

#### Risk Tier Assignment

```
ARI Score Range    Risk Tier    Map Color
─────────────────────────────────────────
[0.00 – 0.30)      Low          Green
[0.30 – 0.50)      Moderate     Yellow
[0.50 – 0.70)      Severe       Orange
[0.70 – 1.00]      Critical     Red
```

#### ARI computation flow

```
cluster_data.joblib (12 clusters)
feature_importances.joblib
        │
        ├─ Compute W1, W2, W3 from importances (normalize to sum=1)
        │
        ├─ For each cluster:
        │    Severity_Score = (Mean_Severity - global_min) / (global_max - global_min)
        │    Density_Score  = Incident_Count / max_count_across_all_clusters
        │    Env_Score      = WEATHER_RISK_MAP[Dominant_Weather]
        │    ARI_Score      = W1 × Severity + W2 × Density + W3 × Env
        │    Risk_Tier      = tier from thresholds table above
        │
        └─ Save:
             ari_data.joblib      → cluster DataFrame + ARI_Score + Risk_Tier columns
             eda/ari_results.json → [{ Cluster_ID, Centroid_Lat, Centroid_Lon,
                                        Incident_Count, Mean_Severity, ARI_Score,
                                        Risk_Tier, Dominant_Weather, Dominant_Area }]
```

---

## 8. Flow 3 — Prediction Request

### What happens when a user submits the prediction form

```
Predict.jsx
│
│  User fills form:
│    cluster_id   = 0
│    hour         = 18
│    day_of_week  = 4   (Friday)
│    is_night     = 1
│    weather      = "Rain"
│    light_cond   = "Darkness - lights lit"
│    road_surface = "Wet or damp"
│    collision    = "Rear-end"
│    cause        = "Overspeed"
│    experience   = "5-10yr"
│    age_band     = "18-30"
│    num_vehicles = 2
│
│  POST /api/predict  ──────────────────────────────────────────────►
│                                                                    │
│                                                              routes/predictions.py
│                                                              predict_severity()
│                                                                    │
│                                                              ├─ Load from memory:
│                                                              │    rf_model.joblib
│                                                              │    label_encoders.joblib
│                                                              │    ari_data.joblib
│                                                              │    feature_importances.joblib
│                                                              │
│                                                              ├─ Encode string inputs:
│                                                              │    weather → label_encoders["Weather_Binned"].transform(["Rain"])
│                                                              │    light   → label_encoders["Light_conditions"].transform([...])
│                                                              │    etc.
│                                                              │
│                                                              ├─ Build feature vector:
│                                                              │    [hour, day_of_week, is_night,
│                                                              │     weather_enc, light_enc, road_enc,
│                                                              │     road_type_enc, road_align_enc,
│                                                              │     junction_enc, lanes_enc,
│                                                              │     collision_enc, num_vehicles,
│                                                              │     vehicle_type_enc, cause_enc,
│                                                              │     experience_enc, age_enc,
│                                                              │     cluster_id]
│                                                              │
│                                                              ├─ rf_model.predict([vector])
│                                                              │    → [2]   (Serious Injury)
│                                                              │
│                                                              ├─ rf_model.predict_proba([vector])
│                                                              │    → [[0.14, 0.61, 0.25]]
│                                                              │    → {1: 0.14, 2: 0.61, 3: 0.25}
│                                                              │
│                                                              ├─ Look up cluster ARI from ari_data:
│                                                              │    cluster_id=0 → ARI=0.70, tier="Critical"
│                                                              │
│                                                              └─► 200 {
│                                                                    predicted_severity: 2,
│                                                                    severity_label: "Serious Injury",
│                                                                    severity_probabilities: {1:0.14, 2:0.61, 3:0.25},
│                                                                    ari_score: 0.70,
│                                                                    risk_tier: "Critical",
│                                                                    weights: {W1:0.54, W2:0.39, W3:0.07}
│                                                                  }
◄──────────────────────────────────────────────────────────────────
│
│  UI renders result card:
│    "Serious Injury" (orange)
│    Probabilities: Slight 14% | Serious 61% | Fatal 25%
│    ARI: 0.70 | Risk Tier: Critical (red badge)
│    Weights: Severity 54%, Density 39%, Environment 7%
```

---

## 9. Flow 4 — Frontend Visualization Flows

### Dashboard Flow

```
Dashboard.jsx mounts
        │
        ├─ fetchClusters()    GET /api/clusters?format=json
        │    → Array of cluster objects
        │    Computes: totalClusters, criticalCount, totalIncidents, avgARI
        │    Renders: 4 StatCard tiles
        │
        ├─ Recharts BarChart: top 6 clusters, x=Cluster_ID, y=ARI_Score
        │
        ├─ Recharts PieChart: risk tier distribution
        │    segments: Low (green), Moderate (yellow), Severe (orange), Critical (red)
        │
        └─ Table: first 10 clusters
             columns: ID | Incidents | ARI Score | Risk Tier | Coordinates
```

### Map View Flow

```
MapView.jsx mounts
        │
        ├─ fetchClusters()  GET /api/clusters (GeoJSON FeatureCollection)
        │    Each Feature has:
        │      geometry.coordinates: [lon, lat]
        │      properties: { Cluster_ID, Incident_Count, Mean_Severity,
        │                    ARI_Score, Risk_Tier, Dominant_Weather, ... }
        │
        ├─ React-Leaflet MapContainer, center=(20.59, 78.96), zoom=5
        │    TileLayer: CartoDB Positron (light) or Dark Matter (dark theme)
        │
        ├─ For each cluster feature:
        │    <CircleMarker
        │      center={[lat, lon]}
        │      radius={Math.sqrt(Incident_Count) * 0.5}  ← size ∝ sqrt(count)
        │      fillColor={TIER_COLORS[Risk_Tier]}
        │    >
        │      <Popup> Cluster details card </Popup>
        │    </CircleMarker>
        │
        ├─ Filter buttons: All | Low | Moderate | Severe | Critical
        │    Filters markers client-side by Risk_Tier property
        │
        └─ fitBounds() called after markers render to zoom map to data extent
```

### Analytics Flow

```
Analytics.jsx mounts
        │
        ├─ Parallel fetches (10 calls via Promise.all):
        │    fetchEda("hourly")         → hourly.json
        │    fetchEda("weekly")         → weekly.json
        │    fetchEda("severity")       → severity.json
        │    fetchEda("weather")        → weather.json
        │    fetchEda("top_areas")      → top_areas.json
        │    fetchEda("collision_types") → collision_types.json
        │    fetchEda("causes")         → causes.json
        │    fetchEda("vehicle_types")  → vehicle_types.json
        │    fetchEda("severity_by_weather") → severity_by_weather.json
        │    fetchEda("severity_by_light")   → severity_by_light.json
        │    GET /api/model/metrics     → model_metrics.json
        │
        └─ Renders:
             Model accuracy card (84.54%)
             Hourly LineChart (x=hour, y=count)
             Weekly BarChart (x=day, y=count)
             Severity PieChart
             Weather BarChart
             Top Areas BarChart (horizontal)
             Cross-tab stacked BarChart: Severity × Weather
             Cross-tab stacked BarChart: Severity × Light
             Collision Types BarChart
             Causes RadarChart (polygon for top causes)
```

---

## 10. Data Cleaning — In-Depth

This section consolidates every transformation applied to the raw data, with the rationale for each decision.

### Raw data problems and solutions

| Problem | Column(s) | Solution |
|---|---|---|
| Extra whitespace in values | All string columns | `str.strip()` on all object columns |
| String severity labels | Accident_severity | Map to integers: Slight→1, Serious→2, Fatal→3 |
| Missing categoricals | 14 categorical cols | Fill with `"Unknown"` string |
| Missing vehicle count | Number_of_vehicles_involved | Fill with median to avoid skewing |
| Inconsistent time format | Time | Try HH:MM, HH.MM, decimal; fallback=0 |
| No lat/lng in data | Area_accident_occured | Deterministic geocoding via area→city lookup table |
| All same lat/lng per area | Latitude, Longitude | Add seeded random jitter (±0.015°) so DBSCAN can distinguish records |
| Categorical strings in ML | 14 columns | sklearn LabelEncoder per column |
| Imbalanced severity classes | Accident_severity | `class_weight='balanced'` in Random Forest |
| No temporal feature | Time | Extract Hour (0–23) from Time column |
| No light-based feature | Light_conditions | Binary Is_Night derived from string matching |
| Too many weather values | Weather_conditions | Bin into 6 categories (Clear/Rain/Fog/Snow/Wind/Other) |

### Why deterministic jitter matters

The raw dataset uses area labels like "Residential areas" for all accidents in a residential zone — there are no actual GPS coordinates per accident. Without jitter, every accident in "Residential areas" would have the exact same coordinate (19.076°N, 72.878°E), and DBSCAN with `min_samples=15` would create a single point (a degenerate cluster) rather than a spatial spread.

The jitter uses `hash(area_name) % 1000` as a seed, which means:
- Same area name → same seed → reproducible jitter pattern
- Different rows in same area get different jitter (via `np.random` sequence)
- Re-running the pipeline on the same input produces the same output

---

## 11. Database & Storage Schema

### File-based storage (default)

```
backend/
├── data/
│   ├── road.csv                    Raw CSV (overwritten on each upload)
│   ├── processed_accidents.csv     39-column cleaned + encoded + clustered data
│   ├── upload_history.json         Array of upload metadata objects
│   └── eda/
│       ├── hourly.json             [{hour:int, count:int}] × 24
│       ├── weekly.json             [{day:str, count:int}] × 7
│       ├── severity.json           [{label:str, count:int}] × 3
│       ├── weather.json            [{weather:str, count:int}]
│       ├── top_areas.json          [{area:str, count:int}] (top 20)
│       ├── collision_types.json    [{collision_type:str, count:int}]
│       ├── causes.json             [{cause:str, count:int}]
│       ├── vehicle_types.json      [{vehicle_type:str, count:int}]
│       ├── severity_by_weather.json  [{weather:str, Slight:int, Serious:int, Fatal:int}]
│       ├── severity_by_light.json    [{light:str, Slight:int, Serious:int, Fatal:int}]
│       ├── summary.json            {total_records, severity_distribution, lat/lon ranges}
│       ├── model_metrics.json      {accuracy, confusion_matrix, classification_report, feature_importances}
│       └── ari_results.json        [{Cluster_ID, ARI_Score, Risk_Tier, ...}] × 12
│
└── models/
    ├── label_encoders.joblib       Dict[str, LabelEncoder] — one per encoded column
    ├── dbscan_labels.joblib        np.ndarray (12316,) — cluster label per accident
    ├── cluster_data.joblib         pd.DataFrame (12 rows) — cluster summaries
    ├── rf_model.joblib             sklearn.RandomForestClassifier — trained model
    ├── feature_importances.joblib  Dict[str, float] — feature name → importance
    └── ari_data.joblib             pd.DataFrame — cluster_data + ARI_Score + Risk_Tier
```

### MySQL schema (optional, for scaling)

```sql
-- Database
CREATE DATABASE accident_hotspot_db;

-- Spatial cluster "Black Spots"
CREATE TABLE tbl_Spatial_Clusters (
    Cluster_ID       INT AUTO_INCREMENT PRIMARY KEY,
    Centroid_Lat     DECIMAL(10,8) NOT NULL,
    Centroid_Lon     DECIMAL(11,8) NOT NULL,
    Radius_Eps       DECIMAL(5,2),
    Incident_Count   INT,
    INDEX idx_coords (Centroid_Lat, Centroid_Lon)
);

-- Individual accident records
CREATE TABLE tbl_Accident_Records (
    Record_ID        INT AUTO_INCREMENT PRIMARY KEY,
    Latitude         DECIMAL(10,8),
    Longitude        DECIMAL(11,8),
    Timestamp        DATETIME,
    Weather_Cond     VARCHAR(50),
    Severity_Hist    INT,                          -- 1/2/3
    Cluster_ID       INT,
    FOREIGN KEY (Cluster_ID) REFERENCES tbl_Spatial_Clusters(Cluster_ID),
    INDEX idx_coords (Latitude, Longitude)
);

-- ARI risk assessments per cluster
CREATE TABLE tbl_Risk_Assessments (
    Assessment_ID    INT AUTO_INCREMENT PRIMARY KEY,
    Cluster_ID       INT,
    Pred_Severity    FLOAT,
    ARI_Score        FLOAT,
    Risk_Tier        VARCHAR(20),
    Env_Modifier     VARCHAR(50),
    FOREIGN KEY (Cluster_ID) REFERENCES tbl_Spatial_Clusters(Cluster_ID)
);
```

---

## 12. API Reference

**Base URL**: `http://localhost:5000`  
**Content-Type**: `application/json` for all requests/responses unless noted.

### Health

```
GET /api/health
Response: {
  "status": "ok",
  "models_loaded": {
    "rf_model": true,
    "label_encoders": true,
    "cluster_data": true,
    "ari_data": true
  }
}
```

### Clusters

```
GET /api/clusters
  ?format=geojson  (default) → GeoJSON FeatureCollection
  ?format=json               → plain JSON array

  GeoJSON Feature shape:
  {
    "type": "Feature",
    "geometry": { "type": "Point", "coordinates": [lon, lat] },
    "properties": {
      "Cluster_ID": 0,
      "Incident_Count": 2060,
      "Mean_Severity": 1.78,
      "Dominant_Weather": "Clear",
      "Dominant_Area": "Residential areas",
      "ARI_Score": 0.70,
      "Risk_Tier": "Critical",
      "Radius_Eps": 0.005
    }
  }

GET /api/clusters/<id>
  Response: {
    "cluster": { ...same properties as above... },
    "accidents": [ ...sample accident records in cluster... ]
  }
```

### Predictions

```
POST /api/predict
  Body (required fields): cluster_id, hour, weather
  Body (optional fields): day_of_week, is_night, light_cond,
                           road_surface, collision_type, cause,
                           driving_experience, age_band, num_vehicles

  Response: {
    "predicted_severity": 2,
    "severity_label": "Serious Injury",
    "severity_probabilities": { "1": 0.14, "2": 0.61, "3": 0.25 },
    "ari_score": 0.70,
    "risk_tier": "Critical",
    "weights": { "W1": 0.54, "W2": 0.39, "W3": 0.07 }
  }
```

### EDA Endpoints

```
GET /api/eda/hourly               → [{hour, count}] × 24
GET /api/eda/weekly               → [{day, count}] × 7
GET /api/eda/severity             → [{label, count}] × 3
GET /api/eda/weather              → [{weather, count}]
GET /api/eda/top_areas            → [{area, count}] (top 20)
GET /api/eda/collision_types      → [{collision_type, count}]
GET /api/eda/causes               → [{cause, count}]
GET /api/eda/vehicle_types        → [{vehicle_type, count}]
GET /api/eda/severity_by_weather  → [{weather, Slight, Serious, Fatal}]
GET /api/eda/severity_by_light    → [{light, Slight, Serious, Fatal}]
GET /api/eda/summary              → {total_records, severity_distribution, ...}
GET /api/model/metrics            → {accuracy, confusion_matrix, classification_report,
                                     feature_importances}
```

### Upload & Pipeline

```
POST /api/upload
  Content-Type: multipart/form-data
  Field: "file" (CSV file)
  Response: { "id": "abc123", "filename": "road.csv", "size_bytes": 2048740 }

POST /api/pipeline/run
  Response: {
    "records": 12316,
    "clusters": 12,
    "accuracy": 84.54,
    "ari_range": [0.28, 0.82]
  }

GET /api/uploads
  Response: [ ...upload history array, newest first... ]

DELETE /api/uploads/<id>
  Response: { "deleted": true }

DELETE /api/uploads/<id>?clear_artifacts=true
  Response: { "deleted": true, "artifacts_cleared": true }
```

---

## 13. File-by-File Reference

### Backend files

| File | Purpose | Key functions |
|---|---|---|
| `app.py` | Flask app factory, registers blueprints, loads models into `app.config` on startup | `create_app()` |
| `config.py` | All constants: file paths, DBSCAN params, RF params, ARI tiers, severity map, MySQL config | Module-level constants |
| `run_pipeline.py` | CLI entry point, calls all 5 stages in order, prints timing | `main()` |
| `research_evaluation.py` | Generates all figures, tables, and metrics for IEEE paper | `generate_paper_figures()`, `evaluate_model()` |
| `scripts/preprocess.py` | Loads raw CSV, cleans, geocodes, engineers features, label-encodes | `run()` → `pd.DataFrame` |
| `scripts/eda.py` | Computes 11 aggregations from processed data, writes JSON | `run()` |
| `scripts/clustering.py` | Runs DBSCAN, computes cluster summaries, saves artifacts | `run()` → `pd.DataFrame` |
| `scripts/classifier.py` | Trains Random Forest, evaluates, saves model + metrics | `run()` → `(clf, importances)` |
| `scripts/ari.py` | Derives weights from importances, computes ARI per cluster | `run()` → `pd.DataFrame` |
| `scripts/seed_db.py` | Reads joblib artifacts, inserts rows into MySQL tables | `seed()` |
| `routes/clusters.py` | Flask blueprint: serves cluster data as GeoJSON or JSON | `get_clusters()`, `get_cluster()` |
| `routes/predictions.py` | Flask blueprint: encodes inputs, runs RF prediction, returns result | `predict_severity()` |
| `routes/eda.py` | Flask blueprint: serves pre-computed EDA JSON files | `get_eda(type)`, `get_model_metrics()` |
| `routes/uploads.py` | Flask blueprint: handles file upload, history CRUD | `upload_csv()`, `list_uploads()`, `delete_upload()` |
| `routes/pipeline.py` | Flask blueprint: orchestrates all 5 pipeline stages | `run_pipeline()` |
| `utils/geojson_utils.py` | Converts cluster DataFrame to GeoJSON FeatureCollection | `clusters_to_geojson(df)` |
| `utils/db.py` | MySQL connection pool, schema creation, query helpers | `get_connection()`, `bootstrap_schema()` |

### Frontend files

| File | Purpose | Key state / hooks |
|---|---|---|
| `main.jsx` | ReactDOM.render entry point, wraps app in `<ThemeProvider>` | — |
| `App.jsx` | Top-level layout: sidebar + `<Routes>`. Handles theme toggle | `useTheme()` |
| `api.js` | All Axios calls. Single axios instance with `/api` baseURL | — |
| `ThemeContext.jsx` | Context + provider for dark/light mode. Persists to localStorage | `useTheme()` hook |
| `index.css` | CSS custom properties (`--clr-bg`, `--clr-text`, etc.), global resets | — |
| `pages/Dashboard.jsx` | Fetches clusters + summary, shows KPI tiles + summary charts | `clusters`, `summary` state |
| `pages/MapView.jsx` | Leaflet map with filtered cluster markers | `clusters`, `filter` state |
| `pages/Analytics.jsx` | Fetches 10+ EDA endpoints, renders chart gallery | per-chart data states |
| `pages/Predict.jsx` | Prediction form, calls POST /api/predict, renders result card | `formData`, `result`, `loading` state |
| `pages/DataManager.jsx` | Upload form, pipeline trigger, history table | `uploads`, `pipelineStatus` state |
| `components/StatCard.jsx` | Presentational: icon + label + value + subtitle tile | `icon`, `label`, `value`, `subtitle` props |
| `components/ChartCard.jsx` | Presentational: wrapper div with title for charts | `title`, `children` props |
| `components/RiskBadge.jsx` | Presentational: colored span for risk tier | `tier` prop |
| `components/Loader.jsx` | Animated spinner (CSS animation) | — |

---

## 14. Configuration Reference

All backend configuration lives in `backend/config.py`:

```python
# File paths
RAW_DATA_PATH          = "backend/data/road.csv"
PROCESSED_DATA_PATH    = "backend/data/processed_accidents.csv"
EDA_OUTPUT_DIR         = "backend/data/eda"
MODELS_DIR             = "backend/models"

# DBSCAN
DBSCAN_EPS             = 0.005          # radians ≈ 31.9 km
DBSCAN_MIN_SAMPLES     = 15

# Random Forest
RF_N_ESTIMATORS        = 200
RF_MAX_DEPTH           = 20
RF_TEST_SIZE           = 0.20
RF_RANDOM_STATE        = 42

# ARI thresholds
ARI_TIERS = {
    "Low":      (0.00, 0.30),
    "Moderate": (0.30, 0.50),
    "Severe":   (0.50, 0.70),
    "Critical": (0.70, 1.00),
}

# Severity label mapping
SEVERITY_MAP = {
    "Slight Injury":  1,
    "Serious Injury": 2,
    "Fatal Injury":   3,
}

# Weather risk scores (used in ARI Env_Score)
WEATHER_RISK = {
    "Clear": 0.15, "Rain": 0.55, "Fog": 0.75,
    "Snow":  0.85, "Wind": 0.40, "Other": 0.30
}

# MySQL (read from environment variables)
MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST",     "localhost"),
    "user":     os.getenv("MYSQL_USER",     "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "accident_hotspot_db"),
}
```

Frontend configuration in `frontend/vite.config.js`:

```javascript
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:5000'    // Forward all /api/* requests to Flask
    }
  }
})
```

---

## 15. Local Setup Guide

### Prerequisites

- Python 3.10+
- Node.js 20+
- (Optional) MySQL 8.3+ running locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # Starts Flask on http://localhost:5000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                       # Starts Vite on http://localhost:5173
```

### Run the pipeline from CLI (without UI)

```bash
cd backend
source venv/bin/activate
python run_pipeline.py
# Optional flags:
#   --seed-db         also seeds MySQL after pipeline
#   --skip-preprocess skip Stage 1 (use existing processed_accidents.csv)
```

### Environment variables (optional MySQL)

```bash
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=yourpassword
export MYSQL_DATABASE=accident_hotspot_db
```

### Verifying everything works

1. Open `http://localhost:5173`
2. Navigate to **Data** → upload a CSV → click **Run Pipeline**
3. Wait ~60 seconds for pipeline to complete (watch for green "Completed" badge)
4. Navigate to **Dashboard** → should see 12 clusters with KPI tiles populated
5. Navigate to **Map** → 12 circle markers on India map
6. Navigate to **Analytics** → all charts rendered
7. Navigate to **Predict** → fill form → submit → see severity prediction card

---

*Document generated for AcciHotspot v1.0 — May 2026*
