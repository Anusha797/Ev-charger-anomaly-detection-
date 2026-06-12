# EV Charging Station Anomaly Detection

Unsupervised anomaly detection for EV charging session events.  
Built for the NOC Services Automation take-home assignment.

---

## Project Structure

```
ev_anomaly_detection/
├── data/
│   ├── charging_logs.csv       # Generated synthetic dataset
│   └── ground_truth.csv        # Ground-truth labels (evaluation only)
├── models/                     # Saved model artefacts (created after training)
├── plots/                      # EDA visualisations (created after eda.py)
├── src/
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── features.py             # Preprocessing & feature engineering
│   ├── eda.py                  # Exploratory data analysis
│   └── train.py                # Model training & evaluation
├── predict.py                  # Inference entry point
├── requirements.txt
├── REPORT.md
└── AI_USAGE.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset

```bash
cd ev_anomaly_detection
python src/generate_data.py
```

Creates `data/charging_logs.csv` (~28 800 rows, 10 stations, 60 days).

### 3. Exploratory Data Analysis

```bash
python src/eda.py
```

Saves 6 plots to `plots/`.

### 4. Train the Model

```bash
python src/train.py
```

Trains an Isolation Forest + LOF ensemble and saves artefacts to `models/`.  
Evaluation metrics are printed if `data/ground_truth.csv` is present.

### 5. Run Inference

```bash
python predict.py --input data/charging_logs.csv --output predictions.csv
```

`predictions.csv` is identical to the input file with two extra columns:

| Column | Description |
|--------|-------------|
| `is_anomaly` | `1` if the event is flagged as anomalous, `0` otherwise |
| `anomaly_score` | Continuous anomaly score (higher = more anomalous) |

---

## Approach Summary

| Step | Choice |
|------|--------|
| Missing values | Per-station median imputation |
| Feature engineering | Physics residuals, Z-score vs station baseline, rolling stats, session aggregates |
| Models | Isolation Forest + Local Outlier Factor ensemble |
| Threshold | 95th percentile of training ensemble scores |
| Evaluation | Precision / Recall / F1 / ROC-AUC vs injected ground-truth labels |

See `REPORT.md` for full methodology.
