# AI-Based Accident Hotspot Prediction System — India

A proactive road-safety intelligence platform combining **GIS spatial analysis** with a hybrid ML pipeline (DBSCAN + Random Forest) to identify accident Black Spots across India and predict severity under varying road and environmental conditions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone the Repository](#step-1--clone-the-repository)
4. [Step 2 — Backend Setup](#step-2--backend-setup)
5. [Step 3 — Frontend Setup](#step-3--frontend-setup)
6. [Step 4 — Start the Servers](#step-4--start-the-servers)
7. [Step 5 — Upload Data & Run the Pipeline](#step-5--upload-data--run-the-pipeline)
8. [Verify Everything Works](#verify-everything-works)
9. [Restarting / Resetting](#restarting--resetting)
10. [ML Pipeline Details](#ml-pipeline-details)
11. [API Reference](#api-reference)
12. [Configuration](#configuration)
13. [Project Structure](#project-structure)

---



---

## Architecture Overview

```
road.csv (Road Accident Data)
    │
    ▼  Uploaded via UI → POST /api/upload → POST /api/pipeline/run
┌──────────────────────────────────────────────────────────────────┐
│  ML Pipeline (5 stages)                                          │
│  1. Preprocess  → cleaned CSV + label encoders                   │
│  2. EDA         → 11 JSON stat files                             │
│  3. DBSCAN      → geographic clusters (India cities)             │
│  4. RandomForest→ severity classifier (~84% accuracy)            │
│  5. ARI         → risk scores per cluster                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │  .joblib models + EDA JSON files
                             ▼
                  ┌──────────────────────┐
                  │  Flask REST API      │  ← python app.py (port 5000)
                  └──────────┬───────────┘
                             │  /api proxy (Vite)
                             ▼
                  ┌──────────────────────┐
                  │  React SPA (Vite)    │  ← npm run dev (port 5173)
                  │  Dashboard · Map     │
                  │  Analytics · Predict │
                  │  Data Manager        │
                  └──────────────────────┘
```

---

## Prerequisites

Install these before you begin:

| Tool    | Min version | Check command        | Download                        |
|---------|-------------|----------------------|---------------------------------|
| Python  | 3.10+       | `python --version`   | https://www.python.org/downloads |
| Node.js | 18+         | `node --version`     | https://nodejs.org               |
| npm     | 9+          | `npm --version`      | included with Node.js            |
| git     | any         | `git --version`      | https://git-scm.com              |

> **Windows note:** When installing Python, check **"Add Python to PATH"** in the installer. Use `python` instead of `python3` in all commands below.

---

## Step 1 — Clone the Repository

**Windows (Command Prompt / PowerShell) and macOS/Linux — same command:**

```bash
git clone <your-repo-url>
cd Accident_prediction_system_test-
```

---

## Step 2 — Backend Setup

### Windows — Command Prompt

```bat
cd backend

python -m venv venv

venv\Scripts\activate.bat

pip install -r requirements.txt
```

### Windows — PowerShell

```powershell
cd backend

python -m venv venv

venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run this once first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS / Linux

```bash
cd backend

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

After activation your terminal prompt will show `(venv)` — this means the virtual environment is active. All `pip install` and `python` commands now use the isolated environment.

---

## Step 3 — Frontend Setup

Open a **new terminal** (keep the backend terminal open).

**Windows and macOS/Linux — same commands:**

```bash
cd frontend

npm install
```

---

## Step 4 — Start the Servers

You need **two terminals running at the same time**.

---

### Terminal 1 — Flask API (Backend)

#### Windows — Command Prompt

```bat
cd backend
venv\Scripts\activate.bat
python app.py
```

#### Windows — PowerShell

```powershell
cd backend
venv\Scripts\Activate.ps1
python app.py
```

#### macOS / Linux

```bash
cd backend
source venv/bin/activate
python app.py
```

Expected output (all platforms):
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Confirm the API is healthy:
```bash
curl http://localhost:5000/api/health
```
Expected response (before pipeline runs — `false` until data is uploaded and pipeline is run):
```json
{
  "status": "ok",
  "models_loaded": {
    "ari_data": false,
    "dbscan_labels": false,
    "processed_data": false,
    "random_forest": false
  }
}
```

---

### Terminal 2 — Vite Dev Server (Frontend)

**Windows and macOS/Linux — same command:**

```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

> Vite automatically proxies all `/api/*` requests from port 5173 → port 5000. No CORS configuration needed.

---

## Step 5 — Upload Data & Run the Pipeline

### Option A — Via the UI (recommended)

1. Download the **RTA Dataset** from [Kaggle](https://www.kaggle.com) (search: "Ethiopian Road Traffic Accident Dataset") and save it as `road.csv`

2. Open **http://localhost:5173/data** in your browser

3. Click **Choose File**, select `road.csv`, then click **Upload**
   - A success banner appears and a history entry is created

4. Click **Run Pipeline**
   - All 5 stages run automatically (typically 30–90 seconds)
   - The history entry updates to **Complete** with stats: records, clusters, accuracy, ARI range

5. All pages now show live data:
   - **Dashboard** — cluster KPIs and incident table
   - **Hotspot Map** — circle markers at Indian city coordinates
   - **Analytics** — EDA charts (hourly, weekly, severity, weather, etc.)
   - **Predict** — real-time severity + ARI prediction form

---

### Option B — Via command line

#### Windows — Command Prompt

```bat
cd backend
venv\Scripts\activate.bat
copy C:\path\to\your\road.csv data\road.csv
python run_pipeline.py
```

#### Windows — PowerShell

```powershell
cd backend
venv\Scripts\Activate.ps1
Copy-Item C:\path\to\your\road.csv -Destination data\road.csv
python run_pipeline.py
```

#### macOS / Linux

```bash
cd backend
source venv/bin/activate
cp /path/to/your/road.csv data/road.csv
python run_pipeline.py
```

Pipeline output (all platforms):
```
STEP 1/5 — Preprocessing ...   → data/processed_accidents.csv
STEP 2/5 — EDA ...             → data/eda/*.json
STEP 3/5 — DBSCAN Clustering   → models/dbscan_labels.joblib
STEP 4/5 — Random Forest ...   → models/rf_model.joblib
STEP 5/5 — ARI Scoring ...     → models/ari_data.joblib
```

---

## Verify Everything Works

Run these from any terminal after the pipeline completes:

```bash
# All models should now show true
curl http://localhost:5000/api/health

# Should print "N clusters"
curl http://localhost:5000/api/clusters | python -c "import json,sys; d=json.load(sys.stdin); print(len(d['features']), 'clusters')"

# Upload history
curl http://localhost:5000/api/uploads

# EDA data
curl http://localhost:5000/api/eda/hourly

# Test a prediction
curl -X POST http://localhost:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"cluster_id\": 1, \"hour\": 18, \"day_of_week\": 4, \"weather\": \"Rain\", \"num_vehicles\": 2}"
```

> On macOS/Linux replace the last command's `^` line continuation with `\` and use single quotes around the JSON.

| Page         | URL                              | What to check                                 |
|--------------|----------------------------------|-----------------------------------------------|
| Dashboard    | http://localhost:5173/           | Cluster count, incident counts, ARI scores    |
| Hotspot Map  | http://localhost:5173/map        | Circle markers at Indian city locations       |
| Analytics    | http://localhost:5173/analytics  | EDA charts populated with real data           |
| Predict      | http://localhost:5173/predict    | Cluster dropdown populated; predictions work  |
| Data Manager | http://localhost:5173/data       | History entry shows Complete + metrics        |

---

## Restarting / Resetting

### Restart the backend server

#### Windows — Command Prompt

```bat
:: Kill whatever is running on port 5000
for /f "tokens=5" %a in ('netstat -aon ^| findstr :5000') do taskkill /F /PID %a

cd backend
venv\Scripts\activate.bat
python app.py
```

#### Windows — PowerShell

```powershell
# Kill whatever is running on port 5000
$pid = (netstat -aon | Select-String ":5000").ToString().Trim().Split()[-1]
if ($pid) { taskkill /F /PID $pid }

cd backend
venv\Scripts\Activate.ps1
python app.py
```

#### macOS / Linux

```bash
lsof -ti :5000 | xargs kill -9
cd backend
source venv/bin/activate
python app.py
```

---

### Restart the frontend server

Stop with **Ctrl+C** in the Vite terminal, then:

```bash
cd frontend
npm run dev
```

---

### Full reset — wipe all data and start fresh

#### Windows — Command Prompt

```bat
cd backend

del /q data\road.csv 2>nul
del /q data\processed_accidents.csv 2>nul
del /q data\upload_history.json 2>nul
rmdir /s /q data\eda 2>nul
del /q models\*.joblib 2>nul

for /f "tokens=5" %a in ('netstat -aon ^| findstr :5000') do taskkill /F /PID %a

venv\Scripts\activate.bat
python app.py
```

#### Windows — PowerShell

```powershell
cd backend

Remove-Item -ErrorAction SilentlyContinue data\road.csv
Remove-Item -ErrorAction SilentlyContinue data\processed_accidents.csv
Remove-Item -ErrorAction SilentlyContinue data\upload_history.json
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\eda
Remove-Item -ErrorAction SilentlyContinue models\*.joblib

$pid = (netstat -aon | Select-String ":5000").ToString().Trim().Split()[-1]
if ($pid) { taskkill /F /PID $pid }

venv\Scripts\Activate.ps1
python app.py
```

#### macOS / Linux

```bash
cd backend
rm -f data/road.csv data/processed_accidents.csv data/upload_history.json
rm -rf data/eda
rm -f models/*.joblib
lsof -ti :5000 | xargs kill -9
source venv/bin/activate
python app.py
```

After a full reset go to **http://localhost:5173/data**, upload `road.csv`, and click **Run Pipeline** to rebuild everything.

---

## ML Pipeline Details

### DBSCAN Clustering

| Parameter     | Value     | Notes                                                           |
|---------------|-----------|-----------------------------------------------------------------|
| `eps`         | 0.005 rad | ~32 km radius — groups accidents within a city, separates cities |
| `min_samples` | 15        | Min accidents required to form a Black Spot cluster             |
| `metric`      | haversine | Geographically accurate great-circle distances                  |
| `algorithm`   | ball_tree | Efficient spatial indexing for haversine                        |

### Random Forest features (17)

```
Hour, DayOfWeek, Is_Night, Weather_Binned_Enc, Num_Vehicles,
Type_of_vehicle_Enc, Road_surface_type_Enc, Road_surface_conditions_Enc,
Light_conditions_Enc, Type_of_collision_Enc, Cause_of_accident_Enc,
Road_allignment_Enc, Types_of_Junction_Enc, Lanes_or_Medians_Enc,
Driving_experience_Enc, Age_band_of_driver_Enc, Cluster_ID
```

### Accident Risk Index

```
ARI = W1 × Severity_Score + W2 × Density_Score + W3 × Environmental_Factor
```

| Component            | Derivation                                         |
|----------------------|----------------------------------------------------|
| Severity_Score       | Mean severity in cluster, normalised to [0, 1]     |
| Density_Score        | Cluster incident count / max count across clusters |
| Environmental_Factor | Weather risk score (Clear = 0.15 … Snow = 0.85)   |
| W1, W2, W3           | Derived from RF feature importances (sum to 1)     |

| Risk Tier | ARI Range   | Map colour |
|-----------|-------------|------------|
| Low       | 0.00 – 0.30 | Green      |
| Moderate  | 0.30 – 0.50 | Yellow     |
| Severe    | 0.50 – 0.70 | Orange     |
| Critical  | 0.70 – 1.00 | Red        |

### Geocoding — area type → Indian city

| Area type           | Mapped city     |
|---------------------|-----------------|
| Office areas        | Delhi NCR       |
| Residential areas   | Mumbai          |
| Church areas        | Bangalore       |
| Industrial areas    | Ahmedabad       |
| School areas        | Chennai         |
| Recreational areas  | Kolkata         |
| Hospital areas      | Jaipur          |
| Market areas        | Hyderabad       |
| Rural village areas | Nagpur          |
| Outside rural areas | Varanasi        |
| Other               | Pune            |
| Unknown             | Centre of India |

---

## API Reference

### Core

| Method | Endpoint             | Description                                    |
|--------|----------------------|------------------------------------------------|
| GET    | `/api/health`        | Server and model load status                   |
| GET    | `/api/clusters`      | All clusters as GeoJSON FeatureCollection      |
| GET    | `/api/clusters/<id>` | Single cluster detail                          |
| POST   | `/api/predict`       | Severity + ARI prediction for given conditions |

### EDA

| Method | Endpoint                       | Returns                                         |
|--------|--------------------------------|-------------------------------------------------|
| GET    | `/api/eda/hourly`              | `[{hour, count}]` × 24                         |
| GET    | `/api/eda/weekly`              | `[{day, count}]` × 7                           |
| GET    | `/api/eda/severity`            | `[{label, count}]`                             |
| GET    | `/api/eda/weather`             | `[{weather, count}]`                           |
| GET    | `/api/eda/top_areas`           | `[{area, count}]`                              |
| GET    | `/api/eda/collision_types`     | `[{collision_type, count}]`                    |
| GET    | `/api/eda/causes`              | `[{cause, count}]`                             |
| GET    | `/api/eda/severity_by_weather` | Cross-tab array                                |
| GET    | `/api/model/metrics`           | Accuracy, confusion matrix, feature importances |

### Upload & Pipeline

| Method | Endpoint                                 | Description                                          |
|--------|------------------------------------------|------------------------------------------------------|
| POST   | `/api/upload`                            | Upload CSV (multipart/form-data, field name: `file`) |
| POST   | `/api/pipeline/run`                      | Run full ML pipeline (blocking, ~30–90 s)            |
| GET    | `/api/uploads`                           | Full upload history (newest first)                   |
| DELETE | `/api/uploads/<id>`                      | Remove a history entry only                          |
| DELETE | `/api/uploads/<id>?clear_artifacts=true` | Remove history entry + wipe all model/EDA files      |

### Prediction request body

```json
{
  "cluster_id": 1,
  "hour": 17,
  "day_of_week": 4,
  "is_night": 1,
  "weather": "Rain",
  "num_vehicles": 2,
  "road_surface_conditions": "Wet or damp",
  "light_conditions": "Darkness - lights unlit",
  "type_of_collision": "Vehicle with vehicle collision",
  "cause_of_accident": "No distancing",
  "driving_experience": "5-10yr",
  "age_band_of_driver": "18-30"
}
```

`cluster_id`, `hour`, and `weather` are required. All other fields are optional.

### Prediction response

```json
{
  "predicted_label": "Slight Injury",
  "severity_probabilities": {
    "Slight Injury": 0.8292,
    "Serious Injury": 0.1431,
    "Fatal Injury": 0.0277
  },
  "ari_score": 0.73,
  "risk_tier": "Critical",
  "weights": {
    "W1_severity": 0.54,
    "W2_density": 0.39,
    "W3_environment": 0.07
  }
}
```

---

## Configuration

All config lives in `backend/config.py`. Override with environment variables:

| Variable             | Default | Description                          |
|----------------------|---------|--------------------------------------|
| `DBSCAN_EPS`         | `0.005` | Cluster radius in radians (~32 km)   |
| `DBSCAN_MIN_SAMPLES` | `15`    | Min accidents to form a cluster      |
| `RF_N_ESTIMATORS`    | `200`   | Number of trees in Random Forest     |
| `RF_MAX_DEPTH`       | `20`    | Maximum tree depth                   |
| `RF_TEST_SIZE`       | `0.2`   | Test set fraction (stratified split) |

---

## Project Structure

```
Accident_prediction_system_test-/
├── README.md
├── backend/
│   ├── app.py                       # Flask application entry point
│   ├── config.py                    # Centralised paths & hyperparameters
│   ├── requirements.txt             # Python dependencies
│   ├── run_pipeline.py              # CLI pipeline runner (alternative to UI)
│   ├── data/
│   │   ├── road.csv                 # Raw dataset (upload via UI or place here)
│   │   ├── processed_accidents.csv  # Generated by pipeline step 1
│   │   ├── upload_history.json      # Upload log (auto-created)
│   │   └── eda/                     # Generated EDA JSON files
│   ├── models/                      # Serialised ML artefacts (generated)
│   │   ├── rf_model.joblib
│   │   ├── dbscan_labels.joblib
│   │   ├── cluster_data.joblib
│   │   ├── ari_data.joblib
│   │   ├── label_encoders.joblib
│   │   └── feature_importances.joblib
│   ├── scripts/                     # Pipeline stages
│   │   ├── preprocess.py            # Step 1 — clean + encode
│   │   ├── eda.py                   # Step 2 — stat distributions
│   │   ├── clustering.py            # Step 3 — DBSCAN
│   │   ├── classifier.py            # Step 4 — Random Forest
│   │   └── ari.py                   # Step 5 — risk index
│   ├── routes/                      # Flask API blueprints
│   │   ├── clusters.py
│   │   ├── predictions.py
│   │   ├── eda_routes.py
│   │   └── upload.py                # Upload, pipeline, history endpoints
│   └── utils/
│       └── geojson_utils.py         # Cluster → GeoJSON conversion
└── frontend/
    ├── package.json
    ├── vite.config.js               # /api proxy → port 5000
    └── src/
        ├── api.js                   # Axios wrappers for all API calls
        ├── App.jsx                  # Layout + routing
        ├── ThemeContext.jsx         # Dark / light theme toggle
        ├── components/              # Shared UI components
        └── pages/
            ├── Dashboard.jsx        # KPI overview + cluster table
            ├── MapView.jsx          # Leaflet hotspot map
            ├── Analytics.jsx        # EDA charts
            ├── Predict.jsx          # Real-time severity prediction
            └── DataManager.jsx      # Upload, pipeline, history, CSV guide
```
