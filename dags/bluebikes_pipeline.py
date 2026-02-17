"""
BlueForecast Data Pipeline DAG
Orchestrates the full data pipeline from raw ingestion to feature engineering,
schema validation, and bias detection.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline_tasks import (
    download_raw_data,
    clean_data,
    process_station_metadata,
    process_weather_data,
    process_holiday_calendar,
    aggregate_demand,
    run_feature_engineering,
    validate_schema,
    detect_bias,
)

default_args = {
    "owner": "blueforecast",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="bluebikes_data_pipeline",
    default_args=default_args,
    description="End-to-end data pipeline for Bluebikes demand prediction",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bluebikes", "data-pipeline", "mlops"],
) as dag:

    # Stage 1: Download raw data
    t_download = PythonOperator(
        task_id="download_raw_data",
        python_callable=download_raw_data,
    )

    # Stage 2: Clean raw data
    t_clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    # Stage 3: Parallel enrichment tasks
    t_stations = PythonOperator(
        task_id="process_station_metadata",
        python_callable=process_station_metadata,
    )

    t_weather = PythonOperator(
        task_id="process_weather_data",
        python_callable=process_weather_data,
    )

    t_holidays = PythonOperator(
        task_id="process_holiday_calendar",
        python_callable=process_holiday_calendar,
    )

    # Stage 4: Aggregate demand
    t_aggregate = PythonOperator(
        task_id="aggregate_demand",
        python_callable=aggregate_demand,
    )

    # Stage 5: Feature engineering
    t_features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=run_feature_engineering,
    )

    # Stage 6: Schema validation
    t_validate = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema,
    )

    # Stage 7: Bias detection
    t_bias = PythonOperator(
        task_id="detect_bias",
        python_callable=detect_bias,
    )

    # Define task dependencies
    # download → clean → [stations, weather, holidays] → aggregate → features → validate → bias
    t_download >> t_clean
    t_clean >> [t_stations, t_weather, t_holidays]
    [t_stations, t_weather, t_holidays] >> t_aggregate
    t_aggregate >> t_features
    t_features >> t_validate
    t_validate >> t_bias