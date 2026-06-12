"""
eda.py
------
Exploratory Data Analysis for EV charging logs.
Produces visualisation plots and summary statistics.

Run: python src/eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from features import load_data, impute

DATA_DIR  = "data"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def run_eda():
    print("=== EDA: Loading data ===")
    df = load_data(os.path.join(DATA_DIR, "charging_logs.csv"))

    # ------------------------------------------------------------------ #
    # 1. Dataset Overview
    # ------------------------------------------------------------------ #
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Stations: {df['station_id'].nunique()}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    df = impute(df)

    # ------------------------------------------------------------------ #
    # 2. Numeric Distributions
    # ------------------------------------------------------------------ #
    num_cols = ["voltage", "current", "power_kw", "temperature_c", "energy_kwh"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=60, edgecolor="none", alpha=0.8, color="#4C72B0")
        axes[i].set_title(col, fontsize=13)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
    axes[-1].set_visible(False)
    plt.suptitle("Sensor Reading Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_distributions.png"), dpi=120)
    plt.close()
    print("Saved: plots/01_distributions.png")

    # ------------------------------------------------------------------ #
    # 3. Error Code Frequency
    # ------------------------------------------------------------------ #
    err_counts = df[df["error_code"] != 0]["error_code"].value_counts().reset_index()
    err_counts.columns = ["error_code", "count"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=err_counts, x="error_code", y="count", ax=ax, palette="Reds_r")
    ax.set_title("Error Code Frequency (non-zero only)", fontweight="bold")
    ax.set_xlabel("Error Code")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_error_codes.png"), dpi=120)
    plt.close()
    print("Saved: plots/02_error_codes.png")

    # ------------------------------------------------------------------ #
    # 4. Correlation Heatmap
    # ------------------------------------------------------------------ #
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Sensor Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_correlation.png"), dpi=120)
    plt.close()
    print("Saved: plots/03_correlation.png")

    # ------------------------------------------------------------------ #
    # 5. Per-Station Boxplots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="station_id", y="voltage", ax=axes[0], palette="Set2")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title("Voltage by Station")

    sns.boxplot(data=df, x="station_id", y="temperature_c", ax=axes[1], palette="Set2")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_title("Temperature by Station")

    plt.suptitle("Station-Level Variance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_station_variance.png"), dpi=120)
    plt.close()
    print("Saved: plots/04_station_variance.png")

    # ------------------------------------------------------------------ #
    # 6. Temporal Patterns — Average Power by Hour
    # ------------------------------------------------------------------ #
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")["power_kw"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=hourly, x="hour", y="power_kw", marker="o", ax=ax, color="#DD8452")
    ax.set_title("Average Power Delivered by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Mean Power (kW)")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_hourly_power.png"), dpi=120)
    plt.close()
    print("Saved: plots/05_hourly_power.png")

    # ------------------------------------------------------------------ #
    # 7. Scatter: Power vs Temperature (anomaly signal)
    # ------------------------------------------------------------------ #
    # Load ground truth for colour if available
    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, parse_dates=["timestamp"])
        plot_df = df.merge(
            gt[["station_id", "session_id", "timestamp", "_label"]],
            on=["station_id", "session_id", "timestamp"], how="left"
        )
        plot_df["_label"] = plot_df["_label"].fillna(0).astype(int)
        colors = plot_df["_label"].map({0: "#4C72B0", 1: "#C44E52"})
        label_used = True
    else:
        plot_df = df.copy()
        colors = "#4C72B0"
        label_used = False

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["temperature_c"], plot_df["power_kw"],
               c=colors, alpha=0.25, s=8, rasterized=True)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power vs Temperature" + (" (red = fault)" if label_used else ""),
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_power_vs_temp.png"), dpi=120)
    plt.close()
    print("Saved: plots/06_power_vs_temp.png")

    print("\n✅ EDA complete. All plots saved to plots/")


if __name__ == "__main__":
    run_eda()
