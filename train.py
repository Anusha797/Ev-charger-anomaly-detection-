"""
train.py
--------
Trains an anomaly detection ensemble on charging_logs.csv.

Strategy
--------
We use a two-model ensemble:
  1. Isolation Forest  – captures global outliers in feature space efficiently
  2. Local Outlier Factor (LOF, novelty=True) – captures density-based anomalies

Both are unsupervised. Final anomaly score = mean of normalised IF + LOF scores.
Threshold is chosen as the 95th percentile of training scores (flagging ~5% as anomalies).

Outputs
-------
  models/isolation_forest.pkl
  models/lof.pkl
  models/scaler.pkl
  models/encoders.pkl
  models/threshold.txt
  models/station_baselines.pkl   (for inference)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

from features import build_features, FEATURE_COLS

# ------------------------------------------------------------------ #
MODEL_DIR = "models"
DATA_DIR = "data"
CONTAMINATION = 0.05   # expected fraction of anomalies
RANDOM_STATE = 42
# ------------------------------------------------------------------ #


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Loading & Engineering Features ===")
    df, X, encoders = build_features(
        filepath=os.path.join(DATA_DIR, "charging_logs.csv")
    )

    print(f"Feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------ #
    # Scale features
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------ #
    # Train Isolation Forest
    # ------------------------------------------------------------------ #
    print("\n=== Training Isolation Forest ===")
    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_scaled)
    # score_samples returns negative mean anomaly depth; lower = more anomalous
    iso_scores_raw = -iso.score_samples(X_scaled)  # flip so higher = more anomalous

    # ------------------------------------------------------------------ #
    # Train Local Outlier Factor (novelty mode so we can call predict later)
    # ------------------------------------------------------------------ #
    print("=== Training Local Outlier Factor ===")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_scaled)
    lof_scores_raw = -lof.score_samples(X_scaled)  # flip: higher = more anomalous

    # ------------------------------------------------------------------ #
    # Ensemble: min-max normalise each score then average
    # ------------------------------------------------------------------ #
    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    ensemble_score = (minmax(iso_scores_raw) + minmax(lof_scores_raw)) / 2

    # ------------------------------------------------------------------ #
    # Threshold: 95th percentile of training ensemble scores
    # ------------------------------------------------------------------ #
    threshold = float(np.percentile(ensemble_score, 95))
    print(f"\nAnomaly threshold (95th pct): {threshold:.4f}")

    df["anomaly_score"] = ensemble_score
    df["is_anomaly"] = (ensemble_score >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    # Evaluate against ground truth (if available)
    # ------------------------------------------------------------------ #
    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    if os.path.exists(gt_path):
        print("\n=== Evaluation Against Ground Truth ===")
        gt = pd.read_csv(gt_path, parse_dates=["timestamp"])
        # Merge on station_id + session_id + timestamp
        merged = df.merge(
            gt[["station_id", "session_id", "timestamp", "_label"]],
            on=["station_id", "session_id", "timestamp"],
            how="left",
        )
        merged["_label"] = merged["_label"].fillna(0).astype(int)
        y_true = merged["_label"].values
        y_pred = merged["is_anomaly"].values
        y_score = merged["anomaly_score"].values

        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
        try:
            auc = roc_auc_score(y_true, y_score)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Save models and artefacts
    # ------------------------------------------------------------------ #
    print("\n=== Saving Models ===")
    with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(MODEL_DIR, "lof.pkl"), "wb") as f:
        pickle.dump(lof, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Save station baselines (needed at inference time)
    station_baselines = df[["station_id"]].drop_duplicates().copy()
    with open(os.path.join(MODEL_DIR, "station_baselines.pkl"), "wb") as f:
        pickle.dump(df, f)          # save full training df for reference baselines

    with open(os.path.join(MODEL_DIR, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    print("Saved: isolation_forest.pkl, lof.pkl, scaler.pkl, encoders.pkl, threshold.txt")
    print("\n✅ Training complete.")
    return df, iso, lof, scaler, encoders, threshold


if __name__ == "__main__":
    train()
