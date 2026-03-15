"""
Data Preprocessing Pipeline  —  Indian Road Accident Dataset
==============================================================
Loads road.csv from the Kaggle "Road Accident Severity in India" dataset,
cleans it, geocodes the Area_accident_occured column to approximate lat/lng,
engineers features, encodes categoricals, and writes a processed CSV.
"""

import os, sys, re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, LABEL_ENCODERS_PATH,
    MODELS_DIR, SEVERITY_MAP,
)

# ─────────────────────────────────────────────────────────────────────────────
# Geocoding lookup — maps Area_accident_occured values to approximate
# (latitude, longitude) in India.  Since the raw dataset has NO coordinates,
# we assign representative coords for each known area/locality.  Unknown areas
# get a jittered position near the geographic centre of India.
# ─────────────────────────────────────────────────────────────────────────────
AREA_COORDS = {
    # Major metro areas
    "Office areas":          (28.6280, 77.2190),   # Delhi NCR
    "Residential areas":     (19.0760, 72.8777),   # Mumbai
    "Church areas":          (12.9340, 77.6100),   # Bangalore
    "Industrial areas":      (23.0225, 72.5714),   # Ahmedabad
    "School areas":          (13.0827, 80.2707),   # Chennai
    "Recreational areas":    (22.5726, 88.3639),   # Kolkata
    "Hospital areas":        (26.9124, 75.7873),   # Jaipur
    "Market areas":          (17.3850, 78.4867),   # Hyderabad
    "Rural village areas":   (21.1458, 79.0882),   # Nagpur
    "Outside rural areas":   (25.3176, 82.9739),   # Varanasi
    "Rural village araea":   (21.1458, 79.0882),   # (typo variant in dataset)
    "Outside rural areaas":  (25.3176, 82.9739),   # (typo variant)
    "Unknown":               (20.5937, 78.9629),   # Centre of India
    "Other":                 (18.5204, 73.8567),   # Pune
    "  Other":               (18.5204, 73.8567),
    "  Recreational areas":  (22.5726, 88.3639),
    "  Market areas":        (17.3850, 78.4867),
    "  Office areas":        (28.6280, 77.2190),
}

INDIA_CENTRE = (20.5937, 78.9629)


def _geocode(area: str) -> tuple:
    """Return (lat, lng) for an area name, with random jitter (~2-8 km)."""
    area_clean = str(area).strip()
    base = AREA_COORDS.get(area_clean, AREA_COORDS.get(area_clean.strip(), None))
    if base is None:
        base = INDIA_CENTRE

    rng = np.random.default_rng(hash(area_clean) % (2**31))
    jitter_lat = rng.uniform(-0.06, 0.06)   # ~6 km spread
    jitter_lng = rng.uniform(-0.06, 0.06)
    return (base[0] + jitter_lat, base[1] + jitter_lng)


# ─────────────────────────────────────────────────────────────────────────────
# Weather binning
# ─────────────────────────────────────────────────────────────────────────────
WEATHER_BIN = {
    "Normal": "Clear",
    "Raining": "Rain",
    "Fog or mist": "Fog",
    "Snow": "Snow",
    "Windy": "Wind",
    "Cloudy": "Clear",
    "Other": "Other",
    "Unknown": "Other",
}


def _parse_time_to_hour(t) -> int:
    """Extract hour from the Time column (various formats seen in the dataset)."""
    s = str(t).strip()
    # Try HH:MM:SS or HH.MM.SS
    m = re.match(r"(\d{1,2})[:\.](\d{2})", s)
    if m:
        return int(m.group(1)) % 24
    # Fallback
    try:
        return int(float(s)) % 24
    except (ValueError, TypeError):
        return 12  # noon default


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    print(f"[PREPROCESS] Loading raw CSV from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")
    print(f"  Columns: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[PREPROCESS] Cleaning ...")

    # Standardise Accident_severity
    df["Accident_severity"] = df["Accident_severity"].str.strip()
    df["Severity"] = df["Accident_severity"].map(SEVERITY_MAP)
    unmapped = df["Severity"].isna().sum()
    if unmapped:
        print(f"  Warning: {unmapped} rows with unmapped severity → defaulting to 1")
        df["Severity"] = df["Severity"].fillna(1).astype(int)
    else:
        df["Severity"] = df["Severity"].astype(int)

    # Fill missing categoricals with "Unknown"
    cat_cols = [
        "Day_of_week", "Age_band_of_driver", "Driving_experience",
        "Type_of_vehicle", "Area_accident_occured", "Lanes_or_Medians",
        "Road_allignment", "Types_of_Junction", "Road_surface_type",
        "Road_surface_conditions", "Light_conditions", "Weather_conditions",
        "Type_of_collision", "Vehicle_movement", "Pedestrian_movement",
        "Cause_of_accident",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    # Number_of_vehicles_involved — fill with median
    if "Number_of_vehicles_involved" in df.columns:
        df["Number_of_vehicles_involved"] = pd.to_numeric(
            df["Number_of_vehicles_involved"], errors="coerce"
        )
        df["Number_of_vehicles_involved"] = df["Number_of_vehicles_involved"].fillna(
            df["Number_of_vehicles_involved"].median()
        ).astype(int)

    print(f"  {len(df):,} rows after cleaning.")
    return df


def geocode_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Assign approximate lat/lng from Area_accident_occured."""
    print("[PREPROCESS] Geocoding areas → lat/lng ...")
    np.random.seed(42)
    coords = df["Area_accident_occured"].apply(_geocode)
    df["Latitude"] = coords.apply(lambda c: c[0])
    df["Longitude"] = coords.apply(lambda c: c[1])

    # Add per-row jitter so identical areas spread out (needed for DBSCAN)
    n = len(df)
    df["Latitude"] += np.random.uniform(-0.015, 0.015, n)
    df["Longitude"] += np.random.uniform(-0.015, 0.015, n)

    print(f"  Lat range: [{df['Latitude'].min():.4f}, {df['Latitude'].max():.4f}]")
    print(f"  Lng range: [{df['Longitude'].min():.4f}, {df['Longitude'].max():.4f}]")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[PREPROCESS] Engineering features ...")

    # Hour from Time column
    df["Hour"] = df["Time"].apply(_parse_time_to_hour)

    # Day of week → numeric
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["DayOfWeek"] = df["Day_of_week"].map(day_map).fillna(3).astype(int)

    # Is_Night from Light_conditions
    night_keywords = ["dark", "night"]
    df["Is_Night"] = df["Light_conditions"].str.lower().apply(
        lambda x: 1 if any(k in str(x) for k in night_keywords) else 0
    )

    # Weather binned
    df["Weather_Binned"] = df["Weather_conditions"].map(WEATHER_BIN).fillna("Other")

    # Number of vehicles (ensure int)
    if "Number_of_vehicles_involved" in df.columns:
        df["Num_Vehicles"] = df["Number_of_vehicles_involved"]
    else:
        df["Num_Vehicles"] = 1

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("[PREPROCESS] Encoding categoricals ...")
    encoders = {}
    encode_cols = [
        "Weather_Binned", "Type_of_vehicle", "Road_surface_type",
        "Road_surface_conditions", "Light_conditions", "Type_of_collision",
        "Cause_of_accident", "Area_accident_occured", "Road_allignment",
        "Types_of_Junction", "Lanes_or_Medians", "Driving_experience",
        "Age_band_of_driver", "Vehicle_movement", "Pedestrian_movement",
    ]
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_Enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, LABEL_ENCODERS_PATH)
    print(f"  Saved {len(encoders)} label encoders → {LABEL_ENCODERS_PATH}")
    return df


def save(df: pd.DataFrame) -> None:
    keep_cols = [
        "Severity", "Latitude", "Longitude",
        "Hour", "DayOfWeek", "Is_Night",
        "Weather_Binned", "Weather_Binned_Enc",
        "Num_Vehicles",
        "Type_of_vehicle", "Type_of_vehicle_Enc",
        "Road_surface_type", "Road_surface_type_Enc",
        "Road_surface_conditions", "Road_surface_conditions_Enc",
        "Light_conditions", "Light_conditions_Enc",
        "Type_of_collision", "Type_of_collision_Enc",
        "Cause_of_accident", "Cause_of_accident_Enc",
        "Area_accident_occured", "Area_accident_occured_Enc",
        "Road_allignment", "Road_allignment_Enc",
        "Types_of_Junction", "Types_of_Junction_Enc",
        "Lanes_or_Medians", "Lanes_or_Medians_Enc",
        "Driving_experience", "Driving_experience_Enc",
        "Age_band_of_driver", "Age_band_of_driver_Enc",
        "Vehicle_movement", "Vehicle_movement_Enc",
        "Pedestrian_movement", "Pedestrian_movement_Enc",
    ]
    # Keep only columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[PREPROCESS] Saved processed data → {PROCESSED_DATA_PATH}  ({len(df):,} rows)")


def run():
    df = load(RAW_DATA_PATH)
    df = clean(df)
    df = geocode_areas(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save(df)
    return df


if __name__ == "__main__":
    run()
