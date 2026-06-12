"""
generate_data.py
----------------
Generates a synthetic EV charging session log dataset (charging_logs.csv).
Includes normal behavior, known fault patterns, subtle anomalies, noise,
and missing values.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_STATIONS = 10
N_DAYS = 60
SESSIONS_PER_STATION_PER_DAY = 8
EVENTS_PER_SESSION = 6

ERROR_CODES = {
    0: "No error",
    101: "OverVoltage",
    102: "UnderVoltage",
    201: "OverCurrent",
    301: "OverTemperature",
    401: "CommunicationError",
    501: "GroundFault",
}

LOG_MESSAGES_NORMAL = [
    "Session started",
    "Charging in progress",
    "Power delivery stable",
    "Vehicle handshake OK",
    "Pilot signal OK",
    "Charging complete",
]

LOG_MESSAGES_FAULT = [
    "Voltage spike detected",
    "Current draw exceeded threshold",
    "Thermal limit warning",
    "Communication timeout",
    "Ground fault detected",
    "Emergency stop triggered",
]


def make_station_baseline():
    """Create per-station baseline parameters to model station-level variance."""
    return {
        sid: {
            "voltage_mean": np.random.normal(240, 3),
            "current_mean": np.random.normal(32, 2),
            "temp_idle": np.random.normal(35, 3),
        }
        for sid in [f"STN_{i:03d}" for i in range(1, N_STATIONS + 1)]
    }


def generate_normal_event(station_id, session_id, ts, event_idx, baseline):
    """Generate a normal charging event row."""
    b = baseline[station_id]
    voltage = np.random.normal(b["voltage_mean"], 2)
    current = np.random.normal(b["current_mean"], 1.5)
    temperature = b["temp_idle"] + event_idx * np.random.normal(1.5, 0.3)
    power_kw = round((voltage * current) / 1000, 3)
    duration_sec = int(np.random.normal(300, 30))
    energy_kwh = round(power_kw * (duration_sec / 3600), 4)
    return {
        "station_id": station_id,
        "timestamp": ts,
        "session_id": session_id,
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "power_kw": power_kw,
        "temperature_c": round(temperature, 2),
        "error_code": 0,
        "message": random.choice(LOG_MESSAGES_NORMAL),
        "duration_sec": duration_sec,
        "energy_kwh": energy_kwh,
        "_label": 0,  # ground truth for evaluation (not used in training)
    }


def inject_fault(row, fault_type):
    """Inject a specific fault pattern into an existing event row."""
    row = row.copy()
    row["_label"] = 1

    if fault_type == "overvoltage":
        row["voltage"] = round(np.random.uniform(270, 300), 2)
        row["error_code"] = 101
        row["message"] = "Voltage spike detected"
        row["power_kw"] = round((row["voltage"] * row["current"]) / 1000, 3)

    elif fault_type == "overcurrent":
        row["current"] = round(np.random.uniform(50, 70), 2)
        row["error_code"] = 201
        row["message"] = "Current draw exceeded threshold"
        row["power_kw"] = round((row["voltage"] * row["current"]) / 1000, 3)

    elif fault_type == "overtemp":
        row["temperature_c"] = round(np.random.uniform(80, 100), 2)
        row["error_code"] = 301
        row["message"] = "Thermal limit warning"

    elif fault_type == "comm_error":
        row["error_code"] = 401
        row["message"] = "Communication timeout"
        row["power_kw"] = 0.0
        row["current"] = 0.0
        row["energy_kwh"] = 0.0

    elif fault_type == "zero_power":
        # Subtle: session active but no energy delivered
        row["power_kw"] = 0.0
        row["energy_kwh"] = 0.0
        row["current"] = round(np.random.uniform(0, 0.5), 2)
        row["error_code"] = 0
        row["message"] = "Charging in progress"  # misleading log

    elif fault_type == "temp_power_mismatch":
        # High temperature but unusually low power — subtle anomaly
        row["temperature_c"] = round(np.random.uniform(70, 85), 2)
        row["power_kw"] = round(np.random.uniform(0.1, 1.0), 3)
        row["error_code"] = 0

    return row


def generate_dataset():
    stations = [f"STN_{i:03d}" for i in range(1, N_STATIONS + 1)]
    baseline = make_station_baseline()
    start_date = datetime(2024, 1, 1, 6, 0, 0)

    rows = []
    session_counter = 1

    for day in range(N_DAYS):
        for station_id in stations:
            for _ in range(SESSIONS_PER_STATION_PER_DAY):
                session_id = f"SES_{session_counter:06d}"
                session_counter += 1

                # Random start time within operating hours (6am–10pm)
                hour_offset = np.random.uniform(0, 16 * 3600)
                session_start = start_date + timedelta(days=day, seconds=hour_offset)

                # Decide if this session is anomalous
                is_fault_session = np.random.rand() < 0.10  # ~10% sessions have faults
                fault_type = random.choice([
                    "overvoltage", "overcurrent", "overtemp",
                    "comm_error", "zero_power", "temp_power_mismatch"
                ]) if is_fault_session else None

                for event_idx in range(EVENTS_PER_SESSION):
                    ts = session_start + timedelta(seconds=event_idx * 300)
                    event = generate_normal_event(
                        station_id, session_id, ts, event_idx, baseline
                    )

                    # Inject fault into 1–2 events per fault session
                    if fault_type and event_idx in [2, 3]:
                        event = inject_fault(event, fault_type)

                    rows.append(event)

    df = pd.DataFrame(rows)

    # Add noise: random NaN values (~2%)
    for col in ["voltage", "current", "power_kw", "temperature_c", "energy_kwh"]:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    # Shuffle
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save ground truth separately, drop _label from main file
    df[["station_id", "session_id", "timestamp", "_label"]].to_csv(
        "data/ground_truth.csv", index=False
    )
    df.drop(columns=["_label"]).to_csv("data/charging_logs.csv", index=False)

    print(f"Generated {len(df)} rows across {session_counter-1} sessions.")
    print(f"Saved: data/charging_logs.csv  |  data/ground_truth.csv")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate_dataset()
