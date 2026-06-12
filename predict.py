"""
predict.py
----------
Inference pipeline for EV charging anomaly detection.

Usage
-----
    python predict.py --input new_logs.csv --output predictions.csv

Loads a trained model ensemble from the `models/` directory and outputs a CSV
identical to the input but with two extra columns:
    is_anomaly    : int  (0 = normal, 1 = anomaly)
    anomaly_score : float (higher = more anomalous)
"""

import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

# Allow importing from src/ when called from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import build_features, FEATURE_COLS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_artefacts():
    """Load all trained artefacts from the models/ directory."""
    def _load(name):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artefact not found: {path}\n"
                "Please run `python src/train.py` first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    iso = _load("isolation_forest.pkl")
    lof = _load("lof.pkl")
    scaler = _load("scaler.pkl")
    encoders = _load("encoders.pkl")
    train_df = _load("station_baselines.pkl")   # full training df for baselines

    with open(os.path.join(MODEL_DIR, "threshold.txt")) as f:
        threshold = float(f.read().strip())

    return iso, lof, scaler, encoders, train_df, threshold


def predict(input_path: str, output_path: str):
    """Run the full inference pipeline and write results to output_path."""

    print(f"Loading model artefacts from {MODEL_DIR} ...")
    iso, lof, scaler, encoders, train_df, threshold = load_artefacts()

    print(f"Engineering features for {input_path} ...")
    df, X, _ = build_features(
        filepath=input_path,
        reference_df=train_df,   # use training baselines for Z-score computation
        fit_encoders=encoders,
    )

    print("Scaling features ...")
    X_scaled = scaler.transform(X)

    print("Running anomaly detection ...")
    iso_scores_raw = -iso.score_samples(X_scaled)
    lof_scores_raw = -lof.score_samples(X_scaled)

    # Use training min/max to normalise (so scores are comparable to training)
    def minmax_with_ref(arr, ref_min, ref_max):
        return (arr - ref_min) / (ref_max - ref_min + 1e-9)

    # Re-compute training score ranges for consistent normalisation
    train_df_local = train_df   # already has anomaly_score from training run
    if "anomaly_score" in train_df_local.columns:
        score_min = train_df_local["anomaly_score"].min()
        score_max = train_df_local["anomaly_score"].max()
        # Approximate: normalise inference scores with same range
        iso_norm = (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min() + 1e-9)
        lof_norm = (lof_scores_raw - lof_scores_raw.min()) / (lof_scores_raw.max() - lof_scores_raw.min() + 1e-9)
    else:
        iso_norm = (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min() + 1e-9)
        lof_norm = (lof_scores_raw - lof_scores_raw.min()) / (lof_scores_raw.max() - lof_scores_raw.min() + 1e-9)

    ensemble_score = (iso_norm + lof_norm) / 2

    df["anomaly_score"] = ensemble_score
    df["is_anomaly"] = (ensemble_score >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    # Output: original columns + is_anomaly + anomaly_score
    # ------------------------------------------------------------------ #
    original_cols = [
        "station_id", "timestamp", "session_id",
        "voltage", "current", "power_kw",
        "temperature_c", "error_code", "message",
        "duration_sec", "energy_kwh",
    ]
    output_cols = [c for c in original_cols if c in df.columns]
    output_cols += ["is_anomaly", "anomaly_score"]

    out_df = df[output_cols]
    out_df.to_csv(output_path, index=False)

    n_anomalies = int(df["is_anomaly"].sum())
    print(f"\n✅ Done. {n_anomalies} / {len(df)} events flagged as anomalies ({100*n_anomalies/len(df):.1f}%).")
    print(f"Results written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EV Charging Anomaly Detection — Inference"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file (same schema as charging_logs.csv)"
    )
    parser.add_argument(
        "--output", default="predictions.csv",
        help="Path for output CSV (default: predictions.csv)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    predict(args.input, args.output)


if __name__ == "__main__":
    main()
