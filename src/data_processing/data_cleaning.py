"""
BlueForecast Data Cleaning
Converts raw Bluebikes CSV trip data from GCS into cleaned parquet.

Based on analysis from 01_data_cleaning.ipynb:
- 2023 Jan-Mar use OLD schema (tripduration, starttime...) → skipped
- 2023 Apr-Dec + 2024 all months use NEW schema (ride_id, started_at...)
- Cleaning: dedup, null removal, duration filter, text standardization
- Retention rate: ~99% (7.88M of 7.95M records kept)
"""

import io
import logging
import pandas as pd
from google.cloud import storage

logger = logging.getLogger("bluebikes_pipeline.data_cleaning")
logger.setLevel(logging.INFO)

BUCKET = "bluebikes-demand-predictor-data"

# NEW format columns (Apr 2023 onwards)
EXPECTED_COLS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual"
]

# 2023: only Apr-Dec (new format); 2024: all months
RAW_FILES = {
    "2023": [f"raw/historical/2023/csv/2023{m:02d}-bluebikes-tripdata.csv" for m in range(4, 13)],
    "2024": [f"raw/historical/2024/csv/2024{m:02d}-bluebikes-tripdata.csv" for m in range(1, 13)],
}


def _download_csv(client, blob_path):
    """Download a single CSV from GCS and return as DataFrame."""
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        logger.warning("File not found: %s — skipping", blob_path)
        return None
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data), parse_dates=["started_at", "ended_at"])
    logger.info("  Loaded %s: %d rows", blob_path.split("/")[-1], len(df))
    return df


def _clean_dataframe(df, label="unknown"):
    """Apply all cleaning steps to a DataFrame."""
    initial = len(df)
    logger.info("Cleaning %s — initial: %d rows", label, initial)

    # 1. Remove duplicates by ride_id
    before = len(df)
    df = df.drop_duplicates(subset=["ride_id"])
    logger.info("  Removed %d duplicates", before - len(df))

    # 2. Calculate trip duration in seconds
    df["trip_duration_seconds"] = (
        df["ended_at"] - df["started_at"]
    ).dt.total_seconds()

    # 3. Remove rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=["ride_id", "started_at", "ended_at",
                           "start_station_id", "end_station_id"])
    logger.info("  Removed %d rows with missing critical fields", before - len(df))

    # 4. Filter duration outliers (< 1 min or > 24 hours)
    before = len(df)
    df = df[(df["trip_duration_seconds"] >= 60) &
            (df["trip_duration_seconds"] <= 86400)]
    logger.info("  Removed %d duration outliers", before - len(df))

    # 5. Standardize text fields
    df["rideable_type"] = df["rideable_type"].str.strip().str.lower()
    df["member_casual"] = df["member_casual"].str.strip().str.lower()

    # 6. Add derived time columns
    df["trip_duration_minutes"] = df["trip_duration_seconds"] / 60
    df["start_hour"] = df["started_at"].dt.hour
    df["start_day_of_week"] = df["started_at"].dt.dayofweek + 1  # 1=Mon match Spark
    df["start_month"] = df["started_at"].dt.month
    df["start_year"] = df["started_at"].dt.year

    final = len(df)
    retention = (final / initial) * 100 if initial > 0 else 0
    logger.info("  Cleaning complete: %d → %d (%.1f%% retained)", initial, final, retention)
    return df


def clean_data(**kwargs):
    """
    Airflow-callable: download raw CSVs from GCS, clean, upload parquet back to GCS.
    Reads from: gs://BUCKET/raw/historical/{year}/csv/*.csv
    Writes to:  gs://BUCKET/processed/cleaned/year={year}/cleaned.parquet
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    total_rows = 0

    for year, file_paths in RAW_FILES.items():
        logger.info("=== Processing year %s ===", year)
        frames = []
        for path in file_paths:
            df = _download_csv(client, path)
            if df is not None:
                frames.append(df)

        if not frames:
            logger.warning("No data found for year %s — skipping", year)
            continue

        df_year = pd.concat(frames, ignore_index=True)
        logger.info("Year %s combined: %d rows", year, len(df_year))

        df_clean = _clean_dataframe(df_year, label=year)
        total_rows += len(df_clean)

        # Upload cleaned parquet to GCS
        out_buf = io.BytesIO()
        df_clean.to_parquet(out_buf, index=False)
        out_buf.seek(0)

        out_path = f"processed/cleaned/year={year}/cleaned.parquet"
        blob = bucket.blob(out_path)
        blob.upload_from_file(out_buf, content_type="application/octet-stream")
        logger.info("Uploaded gs://%s/%s (%d rows)", BUCKET, out_path, len(df_clean))

    logger.info("=== Data cleaning complete. Total cleaned rows: %d ===", total_rows)
    return f"Cleaned {total_rows} total rows"