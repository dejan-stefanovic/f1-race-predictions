# F1 Race Outcome Predictions

An end-to-end machine learning pipeline to predict whether a Formula 1 driver will finish in the points (top 10) using 75 seasons of historical race data. The final XGBoost model achieves **90% accuracy**, outperforming a logistic regression baseline by **24 percentage points**.

---

## Overview

Predicting F1 race outcomes is genuinely hard. Points allocations have changed across eras, driver and constructor performance drifts across seasons, and qualifying data is missing for a significant portion of historical races. This project addresses those challenges through careful feature engineering, a hidden Elo rating system, and a Top-K post-processing step that enforces the real-world constraint that exactly K drivers finish in the points per race.

---

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression v1 — naive baseline (no temporal features) | 77% | 0.935 |
| Logistic Regression v2 — with feature engineering | 86% | 0.935 |
| XGBoost + Top-K post-processing (final model) | **90.6%** | **0.964** |

Feature engineering (v1 → v2) accounted for the majority of the improvement (+9pp), with the switch to XGBoost and Top-K post-processing adding a further 4.6pp.

---

## Key Features Engineered

| Feature | Description |
|---------|-------------|
| `driver_elo` | Hidden Elo rating tracking driver skill over career |
| `constructor_elo` | Hidden Elo rating tracking constructor performance |
| `driver_points_ewma` | Exponentially weighted moving average of recent points |
| `driver_finish_trend_5` | Linear trend in finishing positions over last 5 races |
| `constructor_points_rate_5` | Constructor rolling average points over last 5 races |
| `best_qualifying` | Fastest qualifying time across Q1/Q2/Q3 sessions |
| `driver_age` | Driver age at race date |

**Elo rating design:** Ratings decay toward the global mean between seasons (`SEASON_DECAY = 0.85`) to account for driver and team changes. Pairwise updates are position-weighted so finishing 1st vs 2nd carries more signal than finishing 10th vs 11th.

**Top-K post-processing:** Rather than using a fixed probability threshold, the model selects exactly K drivers per race ranked by predicted probability, where K matches the actual number of points finishers. This enforces the real-world constraint and significantly improves precision.

---

## Project Structure

```
f1-race-predictions/
│
├── data/                                  # Raw data (Ergast API)
│   ├── results.csv
│   ├── races.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   ├── qualifying.csv
│   ├── status.csv
│   └── ...
│
├── outputs/                               # Generated predictions and plots
│
├── baseline_logistic_regression.py        # Baseline model with full evaluation
├── f1_race_predictions_xgboost.py         # Final model
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/dejan-stefanovic/f1-race-predictions.git
cd f1-race-predictions
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
source venv/bin/activate        # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the baseline model
```bash
python baseline_logistic_regression.py
```

### 5. Run the final XGBoost model
```bash
python f1_race_predictions_xgboost.py
```

Outputs (predictions CSV and evaluation plots) are saved to the `outputs/` folder.

---

## Data

Data sourced from the [Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) dataset on Kaggle, compiled by Rohan Rao from the Ergast Motor Racing API.

The model trains on all seasons up to and including 2022, and evaluates on 2023 onward.

**Note:** `lap_times.csv` is excluded from this repository due to file size (~17MB). It is available directly from the Kaggle dataset linked above if needed for future feature development.

---

## Methodology

1. **Data merging** — Results, qualifying, driver, constructor, and race metadata joined into a single modelling dataframe
2. **Preprocessing** — Qualifying lap times converted from string format to seconds; driver age computed from race date and date of birth
3. **Feature engineering** — Rolling form metrics computed with `.shift(1)` to prevent data leakage; Elo ratings computed sequentially across races
4. **Train/test split** — Temporal split at 2022 season boundary to simulate real prediction conditions
5. **Pipeline** — Median imputation + standard scaling for numerical features; constant imputation + one-hot encoding for categorical features
6. **Post-processing** — Top-K selection per race enforces real-world points allocation constraints

---

## Author

**Dejan Stefanovic**
[LinkedIn](https://linkedin.com/in/dejan-stefanovic) · [GitHub](https://github.com/dejan-stefanovic)
dejanstef7@gmail.com
