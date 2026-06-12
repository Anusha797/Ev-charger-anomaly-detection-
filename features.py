"""
features.py
-----------
Preprocessing and feature engineering pipeline for EV charging anomaly detection.

Steps:
  1. Parse and clean raw charging_logs.csv
  2. Impute missing values
  3. Engineer time-based, rolling, session-level, and station-level features
  4. Return a clean feature matrix ready for model training / inference
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------------------------------- #
# 1. Load & Basic Parse
# --------------------------------------------------------------------------- #

def load_data(filepath: str) -> pd.DataFrame:
    """Load charging logs CSV and parse timestamps."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values(["station_id", "session_id", "timestamp"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 2. Impute Missing Values
# --------------------------------------------------------------------------- #

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute numeric columns with the per-station median.
    This avoids leaking global statistics and respects station-level baselines.
    """
    numeric_cols = ["voltage", "current", "power_kw", "temperature_c", "energy_kwh", "duration_sec"]
    for col in numeric_cols:
        station_median = df.groupby("station_id")[col].transform("median")
        df[col] = df[col].fillna(station_median).fillna(df[col].median())
    return df


# --------------------------------------------------------------------------- #
# 3. Time-Based Features
# --------------------------------------------------------------------------- #

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar and temporal features from the event timestamp."""
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek      # 0=Mon, 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    # Cyclical encoding for hour (captures 23→0 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# --------------------------------------------------------------------------- #
# 4. Physical Consistency Features
# --------------------------------------------------------------------------- #

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that capture physical relationships between readings.
    Deviations from expected relationships signal instrumentation faults or
    anomalous charging behaviour.
    """
    # Expected power from V*I (kW); deviation = sensor disagreement
    df["expected_power_kw"] = (df["voltage"] * df["current"]) / 1000
    df["power_residual"] = (df["power_kw"] - df["expected_power_kw"]).abs()

    # Energy consistency: energy ~ power * duration (in hours)
    df["expected_energy_kwh"] = df["power_kw"] * (df["duration_sec"] / 3600)
    df["energy_residual"] = (df["energy_kwh"] - df["expected_energy_kwh"]).abs()

    # Power-to-temperature ratio: high temp / low power is suspicious
    df["power_temp_ratio"] = df["power_kw"] / (df["temperature_c"] + 1)

    # Binary: error code present
    df["has_error"] = (df["error_code"] != 0).astype(int)

    return df


# --------------------------------------------------------------------------- #
# 5. Session-Level Aggregates
# --------------------------------------------------------------------------- #

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate statistics per session and merge back to event level.
    Anomalous sessions often show unusual aggregate patterns.
    """
    agg = df.groupby("session_id").agg(
        session_mean_power=("power_kw", "mean"),
        session_max_temp=("temperature_c", "max"),
        session_total_energy=("energy_kwh", "sum"),
        session_error_count=("has_error", "sum"),
        session_n_events=("power_kw", "count"),
        session_voltage_std=("voltage", "std"),
        session_current_std=("current", "std"),
    ).reset_index()
    agg["session_voltage_std"] = agg["session_voltage_std"].fillna(0)
    agg["session_current_std"] = agg["session_current_std"].fillna(0)
    df = df.merge(agg, on="session_id", how="left")
    return df


# --------------------------------------------------------------------------- #
# 6. Station-Level Baseline Features
# --------------------------------------------------------------------------- #

def add_station_features(df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute per-station baseline statistics and express each event as a
    Z-score deviation from its station's normal behaviour.

    If reference_df is provided (training set), use its baselines for inference.
    Otherwise compute baselines from df itself (training mode).
    """
    src = reference_df if reference_df is not None else df

    station_stats = src.groupby("station_id").agg(
        station_mean_voltage=("voltage", "mean"),
        station_std_voltage=("voltage", "std"),
        station_mean_current=("current", "mean"),
        station_std_current=("current", "std"),
        station_mean_temp=("temperature_c", "mean"),
        station_std_temp=("temperature_c", "std"),
        station_mean_power=("power_kw", "mean"),
        station_std_power=("power_kw", "std"),
    ).reset_index()

    df = df.merge(station_stats, on="station_id", how="left")

    # Z-score deviations
    for col in ["voltage", "current", "temperature_c", "power_kw"]:
        mean_col = f"station_mean_{col.replace('temperature_c','temp').replace('power_kw','power')}"
        std_col  = f"station_std_{col.replace('temperature_c','temp').replace('power_kw','power')}"
        df[f"{col}_zscore"] = (df[col] - df[mean_col]) / (df[std_col] + 1e-6)

    return df


# --------------------------------------------------------------------------- #
# 7. Rolling / Lag Features (within session, sorted by timestamp)
# --------------------------------------------------------------------------- #

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling mean and delta (lag-1 diff) for key sensors within each
    session. Rapid changes within a session are a strong anomaly signal.
    """
    df = df.sort_values(["session_id", "timestamp"]).copy()

    for col in ["voltage", "current", "temperature_c", "power_kw"]:
        rolled = (
            df.groupby("session_id")[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        df[f"{col}_roll3"] = rolled

        delta = df.groupby("session_id")[col].transform(lambda x: x.diff().fillna(0))
        df[f"{col}_delta"] = delta

    return df


# --------------------------------------------------------------------------- #
# 8. Encode Categorical Columns
# --------------------------------------------------------------------------- #

def encode_categoricals(df: pd.DataFrame, fit_encoders: dict = None):
    """
    Label-encode station_id. Returns (df, encoders) so the same encoding can
    be applied at inference time.
    """
    encoders = fit_encoders or {}

    if "station_id" not in encoders:
        le = LabelEncoder()
        df["station_id_enc"] = le.fit_transform(df["station_id"].astype(str))
        encoders["station_id"] = le
    else:
        le = encoders["station_id"]
        df["station_id_enc"] = le.transform(df["station_id"].astype(str))

    return df, encoders


# --------------------------------------------------------------------------- #
# 9. Select Final Feature Columns
# --------------------------------------------------------------------------- #

FEATURE_COLS = [
    # Raw sensor readings
    "voltage", "current", "power_kw", "temperature_c",
    "duration_sec", "energy_kwh",
    # Physics residuals
    "power_residual", "energy_residual", "power_temp_ratio",
    # Error flag
    "has_error",
    # Time features
    "hour_sin", "hour_cos", "is_weekend",
    # Z-scores vs station baseline
    "voltage_zscore", "current_zscore", "temperature_c_zscore", "power_kw_zscore",
    # Rolling / delta
    "voltage_roll3", "current_roll3", "temperature_c_roll3", "power_kw_roll3",
    "voltage_delta", "current_delta", "temperature_c_delta", "power_kw_delta",
    # Session aggregates
    "session_mean_power", "session_max_temp", "session_total_energy",
    "session_error_count", "session_voltage_std", "session_current_std",
    # Station encoding
    "station_id_enc",
]


# --------------------------------------------------------------------------- #
# 10. Full Pipeline Entry Point
# --------------------------------------------------------------------------- #

def build_features(
    filepath: str,
    reference_df: pd.DataFrame = None,
    fit_encoders: dict = None,
):
    """
    End-to-end feature pipeline.

    Parameters
    ----------
    filepath : str
        Path to charging_logs.csv (or new_logs.csv for inference).
    reference_df : pd.DataFrame, optional
        Training DataFrame used to compute station baselines at inference time.
    fit_encoders : dict, optional
        Fitted LabelEncoders from training; required at inference time.

    Returns
    -------
    df : pd.DataFrame
        Full DataFrame with all engineered features appended.
    X : np.ndarray
        Feature matrix (rows × FEATURE_COLS).
    encoders : dict
        Fitted encoders (reuse at inference).
    """
    df = load_data(filepath)
    df = impute(df)
    df = add_time_features(df)
    df = add_physics_features(df)
    df = add_session_features(df)
    df = add_station_features(df, reference_df=reference_df)
    df = add_rolling_features(df)
    df, encoders = encode_categoricals(df, fit_encoders=fit_encoders)

    X = df[FEATURE_COLS].fillna(0).values
    return df, X, encoders
