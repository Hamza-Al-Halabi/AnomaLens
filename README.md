# AnomaLens 

> **From Black-Box to Explainability** — Probabilistic Automata for Time Series Anomaly Detection

AnomaLens compares interpretable automata-based models against deep learning (LSTM, GRU, 1D-CNN) 
for anomaly detection on real-world industrial time series datasets.

## Highlights
-  **Two datasets**: SKAB (valve sensor data) and BATADAL (water distribution network attacks)
-  **Deep learning**: LSTM, GRU, 1D-CNN with 5-seed statistical validation
-  **Probabilistic automata**: PAA → SAX → sliding window state transitions
-  **Explainability**: per-decision path probability, confidence scores, state transition traces
-  **Robustness**: Gaussian noise testing + unseen pattern handling via Levenshtein distance
-  **Statistical analysis**: Wilcoxon/McNemar tests, cross-dataset generalization

## Datasets
| Dataset | Task | Rows | Anomaly Rate |
|---------|------|------|--------------|
| SKAB (valve1 + valve2) | Sensor anomaly detection | 22,472 | 34.83% |
| BATADAL (Training Dataset 2) | Cyber-attack detection | 4,177 | 5.24% |