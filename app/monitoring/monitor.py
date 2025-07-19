import pandas as pd
import json
from loguru import logger
import os
from prometheus_client import Gauge, REGISTRY
from scipy.stats import ks_2samp
import time

# Define the Prometheus Gauge for the p-value with error handling for duplicates
try:
    DATA_DRIFT_P_VALUE = Gauge(
        "data_drift_p_value",
        "P-value from KS test for data drift detection",
        ["feature"]
    )
except ValueError as e:
    if "Duplicated timeseries" in str(e):
        # If already registered, get the existing one
        DATA_DRIFT_P_VALUE = None
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, '_name') and collector._name == 'data_drift_p_value':
                DATA_DRIFT_P_VALUE = collector
                break
        if DATA_DRIFT_P_VALUE is None:
            # Create a new registry if needed
            DATA_DRIFT_P_VALUE = Gauge(
                "data_drift_p_value_alt",
                "P-value from KS test for data drift detection",
                ["feature"]
            )
    else:
        raise

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROFILE_PATH = os.path.join(PROJECT_ROOT, "data/monitoring/reference_profile.json")
PRODUCTION_DATA_PATH = os.path.join(PROJECT_ROOT, "data/monitoring/production_data.parquet")

def run_drift_analysis():
    """
    Performs data drift analysis by comparing production data with a reference profile.
    """
    logger.info("Starting data drift analysis...")

    # Load reference profile
    if not os.path.exists(PROFILE_PATH):
        logger.warning("Reference profile not found. Skipping drift analysis.")
        return
    with open(PROFILE_PATH, 'r') as f:
        reference_profile = json.load(f)

    # Load production data
    if not os.path.exists(PRODUCTION_DATA_PATH):
        logger.warning("Production data not found. Skipping drift analysis.")
        return
    df_prod = pd.read_parquet(PRODUCTION_DATA_PATH)

    # Perform KS test for numerical features
    for feature, stats in reference_profile["numerical"].items():
        if feature in df_prod.columns:
            # Ensure the column is numeric
            df_prod[feature] = pd.to_numeric(df_prod[feature], errors='coerce')
            df_prod.dropna(subset=[feature], inplace=True)
            
            # Create a dummy series with the reference distribution
            reference_series = pd.Series(
                pd.np.random.normal(stats["mean"], stats["std"], len(df_prod))
            )
            
            # Perform the KS test
            ks_statistic, p_value = ks_2samp(df_prod[feature], reference_series)
            
            # Update the Prometheus Gauge
            DATA_DRIFT_P_VALUE.labels(feature=feature).set(p_value)
            logger.info(f"Feature '{feature}': p-value = {p_value:.4f}")

    logger.success("Data drift analysis completed.")

def monitoring_job():
    """
    A background job that runs the drift analysis periodically.
    """
    while True:
        run_drift_analysis()
        time.sleep(60)  # Run every 60 seconds
