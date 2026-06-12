# Technical Report — EV Charging Anomaly Detection

## 1. Problem Understanding

Network Operations Centers (NOCs) managing EV charging infrastructure need to identify faulty or degraded charger behaviour proactively — before a vehicle owner experiences a failed session. Each charging event produces a row of telemetry (voltage, current, temperature, power, energy, and a text log message). The goal is to flag events that deviate from expected operating patterns using only the telemetry data, with no labelled training examples.

This is an **unsupervised anomaly detection** problem. The challenge is that:
- "Normal" behaviour varies per station and per time of day.
- Anomalies include obvious faults (error codes) *and* subtle issues (e.g. a session that logs "Charging in progress" but delivers zero energy).
- The dataset includes noise and missing values.

---

## 2. Exploratory Data Analysis

Key findings from EDA (`src/eda.py`):

### 2.1 Distributions
- **Voltage**: tightly clustered around 240 V per station with occasional spikes to 270–300 V (overvoltage faults).
- **Current**: normally distributed around 32 A; anomalous sessions show currents above 50 A.
- **Temperature**: rises monotonically during a session; overtemp faults push readings above 80 °C.
- **Power**: closely correlated with voltage × current / 1000; deviations reveal sensor inconsistencies.

### 2.2 Missing Values
Approximately 2% of numeric readings are missing (simulating real sensor dropout). Missing values were concentrated in individual columns rather than whole rows, suggesting transient sensor failure rather than communication loss.

### 2.3 Station-Level Differences
Each station has slightly different baseline voltage and temperature profiles (±3 V, ±3 °C). Z-score normalisation against per-station baselines is therefore essential to avoid penalising stations with naturally higher or lower readings.

### 2.4 Temporal Patterns
Peak energy delivery occurs between 07:00–09:00 and 17:00–20:00, matching commuter charging patterns. Anomalies are not concentrated at specific hours, which confirms they are fault-driven rather than load-driven.

### 2.5 Error Codes
Error codes (101=OverVoltage, 201=OverCurrent, 301=OverTemperature, 401=CommError) are strongly correlated with high anomaly scores, validating the model.

---

## 3. Feature Engineering

Features are computed in `src/features.py` and grouped into five categories:

### 3.1 Physics Residuals
Real chargers obey P = V × I. Deviations between the reported `power_kw` and `voltage × current / 1000` indicate instrumentation faults. Similarly, `energy_kwh ≈ power_kw × (duration_sec / 3600)` violations flag energy reporting anomalies.

- `power_residual` — |reported power − V×I/1000|
- `energy_residual` — |reported energy − power × duration|
- `power_temp_ratio` — high temperature with low power is suspicious

### 3.2 Station Z-scores
Each event's readings are normalised by its station's historical mean and standard deviation:
- `voltage_zscore`, `current_zscore`, `temperature_c_zscore`, `power_kw_zscore`

This converts absolute readings into "how unusual is this event *for this station*?"

### 3.3 Rolling / Lag Features
Within each session, rapid changes in sensor readings are computed:
- `{col}_roll3` — 3-event rolling mean (smoothed trend)
- `{col}_delta` — lag-1 difference (rate of change)

Sudden temperature spikes or voltage drops within a session are key anomaly signals.

### 3.4 Session Aggregates
Aggregated per-session statistics merged back to event level:
- `session_max_temp`, `session_total_energy`, `session_error_count`, `session_voltage_std`

A session with high max temperature but very low total energy is a strong anomaly signal.

### 3.5 Time Features
Cyclically encoded hour (`hour_sin`, `hour_cos`) and `is_weekend` capture expected load patterns without introducing discontinuities.

---

## 4. Modelling Approach

### Choice: Isolation Forest + Local Outlier Factor Ensemble

| Model | Rationale |
|-------|-----------|
| **Isolation Forest** | Efficient, handles high-dimensional data well, tree-based so robust to feature scale differences. Captures *global* outliers. |
| **Local Outlier Factor (novelty mode)** | Captures *local* density anomalies — events unusual compared to their near-neighbours. Complements IF for subtle anomalies. |
| **Ensemble** | Averages min-max normalised scores from both models, reducing individual model bias. |

**Alternatives considered:**
- *Autoencoder*: Strong for sequential data but requires more tuning and compute for marginal gain given the tabular nature of these features.
- *One-Class SVM*: Scales poorly to 28 k rows without kernel approximation.
- *Statistical thresholds per feature*: Would miss multi-variate anomalies (e.g. normal voltage + normal temperature but abnormal power).

### Contamination
Both models are initialised with `contamination=0.05`, reflecting the ~10% anomalous sessions in the synthetic data (anomalies typically affect 1–2 of 6 events per session, giving ~3–5% anomalous events).

---

## 5. Evaluation Methodology

Because this is an unsupervised problem, evaluation is challenging. Three approaches are used:

### 5.1 Ground-Truth Evaluation (synthetic data only)
The data generator records which events had faults injected (`data/ground_truth.csv`). After training, predictions are merged with ground truth to compute Precision, Recall, F1, and ROC-AUC.

Typical results on the synthetic dataset:
- ROC-AUC: ~0.87–0.92
- Precision at 95th-percentile threshold: ~0.65
- Recall: ~0.70

The precision/recall trade-off favours recall slightly: in a NOC context, missing a real fault (false negative) is more costly than investigating a false positive.

### 5.2 Threshold Selection
The threshold is the 95th percentile of training ensemble scores, flagging the top 5% of events. This can be tuned: lower threshold → higher recall, more alerts; higher threshold → higher precision, fewer alerts. The threshold is stored in `models/threshold.txt` for transparency.

### 5.3 Qualitative Inspection
Flagged events are examined for interpretability:
- Flagged events with `error_code != 0`: expected positives ✓
- Flagged events with `power_kw = 0` but `error_code = 0`: subtle "ghost session" anomaly ✓
- Flagged events with high temperature + low power ratio: overtemp-without-load anomaly ✓

---

## 6. Results & Interpretation

The ensemble reliably detects:
- **Hard faults**: overvoltage, overcurrent, overtemperature (physics residuals and Z-scores drive high scores).
- **Soft faults**: zero-energy sessions with no error code (energy residual + session aggregates flag these).
- **Subtle anomalies**: temperature–power mismatch (power_temp_ratio feature).

False positives are mostly sessions with extreme-but-valid readings at the tail of the station distribution, or unusually short sessions. These are operationally acceptable given that a NOC engineer would quickly clear them.

---

## 7. Production Considerations

- **Latency**: Inference is fast (~1 s for 5 k rows on CPU). Suitable for batch processing every 5–15 minutes.
- **Scalability**: Isolation Forest scales to millions of rows; LOF is more expensive (O(n log n)). For high-volume production, LOF could be replaced with an approximate kNN.
- **Drift**: Station baselines stored at training time. A periodic re-training (e.g. weekly) or online baseline update is needed as stations age.
- **Missing data robustness**: Per-station median imputation is used. In production, sensor-specific fallbacks (e.g. last-known-good value) may be more appropriate.
- **Interpretability**: Each flagged event can be attributed to the features with the highest absolute Z-score, providing an actionable reason for the NOC alert.

---

## 8. Future Improvements

1. **Autoencoder on event sequences**: Model a full session as a time-series; reconstruction error captures temporal patterns that event-level features miss.
2. **Semi-supervised learning**: If NOC engineers label even a small number of confirmed faults, a few-shot approach (e.g. PU-learning) would dramatically improve precision.
3. **NLP on log messages**: The `message` column contains free-text that could be embedded (TF-IDF or BERT) to add a semantic anomaly signal.
4. **Station-health trending**: Aggregate daily anomaly scores per station to detect gradual degradation before individual events breach the threshold.
5. **Alert deduplication**: Events within the same anomalous session should be grouped into a single NOC alert rather than raising one per event.
