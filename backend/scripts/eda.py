"""
Exploratory Data Analysis  —  Indian Road Accident Dataset
============================================================
Generates aggregated statistics from the processed accident data and
persists them as JSON files that the Flask API serves to the frontend.
"""

import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DATA_PATH, EDA_OUTPUT_DIR


def _save_json(data, name: str):
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(EDA_OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")


def hourly_distribution(df: pd.DataFrame):
    counts = df.groupby("Hour").size().reindex(range(24), fill_value=0)
    data = [{"hour": int(h), "count": int(c)} for h, c in counts.items()]
    _save_json(data, "hourly")
    return data


def weekly_distribution(df: pd.DataFrame):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = df.groupby("DayOfWeek").size().reindex(range(7), fill_value=0)
    data = [{"day": day_names[d], "day_idx": int(d), "count": int(c)}
            for d, c in counts.items()]
    _save_json(data, "weekly")
    return data


def severity_distribution(df: pd.DataFrame):
    sev_labels = {1: "Slight Injury", 2: "Serious Injury", 3: "Fatal Injury"}
    counts = df.groupby("Severity").size()
    data = [{"severity": int(s), "label": sev_labels.get(s, str(s)),
             "count": int(c)} for s, c in counts.items()]
    _save_json(data, "severity")
    return data


def weather_distribution(df: pd.DataFrame):
    counts = df.groupby("Weather_Binned").size().sort_values(ascending=False)
    data = [{"weather": w, "count": int(c)} for w, c in counts.items()]
    _save_json(data, "weather")
    return data


def area_distribution(df: pd.DataFrame, n=20):
    counts = df.groupby("Area_accident_occured").size().sort_values(ascending=False).head(n)
    data = [{"area": area, "count": int(c)} for area, c in counts.items()]
    _save_json(data, "top_areas")
    return data


def collision_type_distribution(df: pd.DataFrame):
    if "Type_of_collision" not in df.columns:
        return []
    counts = df.groupby("Type_of_collision").size().sort_values(ascending=False)
    data = [{"collision_type": t, "count": int(c)} for t, c in counts.items()]
    _save_json(data, "collision_types")
    return data


def cause_distribution(df: pd.DataFrame):
    if "Cause_of_accident" not in df.columns:
        return []
    counts = df.groupby("Cause_of_accident").size().sort_values(ascending=False)
    data = [{"cause": t, "count": int(c)} for t, c in counts.items()]
    _save_json(data, "causes")
    return data


def vehicle_type_distribution(df: pd.DataFrame):
    if "Type_of_vehicle" not in df.columns:
        return []
    counts = df.groupby("Type_of_vehicle").size().sort_values(ascending=False)
    data = [{"vehicle_type": t, "count": int(c)} for t, c in counts.items()]
    _save_json(data, "vehicle_types")
    return data


def severity_by_weather(df: pd.DataFrame):
    cross = df.groupby(["Weather_Binned", "Severity"]).size().reset_index(name="count")
    data = cross.to_dict(orient="records")
    for d in data:
        d["Severity"] = int(d["Severity"])
        d["count"] = int(d["count"])
    _save_json(data, "severity_by_weather")
    return data


def severity_by_light(df: pd.DataFrame):
    if "Light_conditions" not in df.columns:
        return []
    cross = df.groupby(["Light_conditions", "Severity"]).size().reset_index(name="count")
    data = cross.to_dict(orient="records")
    for d in data:
        d["Severity"] = int(d["Severity"])
        d["count"] = int(d["count"])
    _save_json(data, "severity_by_light")
    return data


def summary_stats(df: pd.DataFrame):
    data = {
        "total_records": int(len(df)),
        "severity_mean": round(float(df["Severity"].mean()), 2),
        "severity_max": int(df["Severity"].max()),
        "severity_distribution": {
            "slight_injury": int((df["Severity"] == 1).sum()),
            "serious_injury": int((df["Severity"] == 2).sum()),
            "fatal_injury": int((df["Severity"] == 3).sum()),
        },
        "unique_areas": int(df["Area_accident_occured"].nunique())
                        if "Area_accident_occured" in df.columns else 0,
        "lat_range": [round(float(df["Latitude"].min()), 4),
                      round(float(df["Latitude"].max()), 4)],
        "lng_range": [round(float(df["Longitude"].min()), 4),
                      round(float(df["Longitude"].max()), 4)],
    }
    _save_json(data, "summary")
    return data


def run():
    print("[EDA] Loading processed data ...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"  {len(df):,} rows loaded.")

    print("[EDA] Generating distributions ...")
    hourly_distribution(df)
    weekly_distribution(df)
    severity_distribution(df)
    weather_distribution(df)
    area_distribution(df)
    collision_type_distribution(df)
    cause_distribution(df)
    vehicle_type_distribution(df)
    severity_by_weather(df)
    severity_by_light(df)
    summary_stats(df)
    print("[EDA] Done.")


if __name__ == "__main__":
    run()
