"""
F1 Race Outcome Predictions — Baseline Model (Logistic Regression)
 
Predicts whether a driver will finish in the points (top 10) using
logistic regression with feature engineering including Elo ratings,
rolling form metrics, and qualifying performance.
 
This script serves as the baseline model comparison for the final
XGBoost model in f1_race_predictions_xgboost.py.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/"
OUTPUT_PATH = "outputs/"
TRAIN_CUTOFF_YEAR = 2022   # train on seasons up to and including this year
RANDOM_STATE = 42

# ── Load Data ─────────────────────────────────────────────────────────────────

results = pd.read_csv(DATA_PATH + "results.csv").drop(
    columns=["milliseconds", "positionText"]
)
races        = pd.read_csv(DATA_PATH + "races.csv")
drivers      = pd.read_csv(DATA_PATH + "drivers.csv")
constructors = pd.read_csv(DATA_PATH + "constructors.csv")
qualifying   = pd.read_csv(DATA_PATH + "qualifying.csv")
status       = pd.read_csv(DATA_PATH + "status.csv")

# ── Merge ─────────────────────────────────────────────────────────────────────

results = (
    results
    .merge(races[["raceId", "year", "round", "circuitId", "name", "date"]], on="raceId", how="left")
    .merge(drivers[["driverId", "driverRef", "dob", "nationality"]], on="driverId", how="left")
    .merge(constructors[["constructorId", "constructorRef", "nationality"]], on="constructorId", how="left")
    .merge(qualifying[["raceId", "driverId", "constructorId", "q1", "q2", "q3"]],
           on=["raceId", "driverId", "constructorId"], how="left")
    .merge(status[["statusId", "status"]], on="statusId", how="left")
    .rename(columns={
        "nationality_x": "driver_nationality",
        "nationality_y": "constructor_nationality",
        "name": "circuit_name"
    })
)

# ── Preprocessing ─────────────────────────────────────────────────────────────

results["date"] = pd.to_datetime(results["date"])
results["dob"]  = pd.to_datetime(results["dob"])
results["driver_age"] = (results["date"] - results["dob"]).dt.days // 365
 
results[["q1", "q2", "q3"]] = results[["q1", "q2", "q3"]].replace(r"\\N", pd.NA)

def lap_time_to_seconds(t):
    """Convert lap time string (M:SS.mmm) to total seconds."""
    if pd.isna(t) or isinstance(t, (int, float)):
        return t if isinstance(t, (int, float)) else pd.NA
    if ":" not in t:
        return pd.NA
    try:
        mins, secs = t.split(":")
        return int(mins) * 60 + float(secs)
    except Exception:
        return pd.NA

for col in ["q1", "q2", "q3"]:
    results[col] = results[col].apply(lap_time_to_seconds)
 
results["best_qualifying"] = results[["q1", "q2", "q3"]].min(axis=1)

# ── Feature Engineering ───────────────────────────────────────────────────────

# Constructor rolling points form (5-race window)
results = results.sort_values(by=["constructorId", "year", "round"])
results["constructor_points_rate_5"] = (
    results.groupby("constructorId")["points"]
    .shift(1)
    .rolling(window=5, min_periods=1)
    .mean()
)
 
# Driver form features
results = results.sort_values(by=["driverId", "year", "round"])
 
results["driver_points_ewma"] = results.groupby("driverId")["points"].transform(
    lambda s: s.shift(1).ewm(span=5, adjust=False).mean()
)
results["driver_last_finish"] = results.groupby("driverId")["positionOrder"].shift(1)
results["driver_last_points"] = results.groupby("driverId")["points"].shift(1)
results["driver_finish_trend_5"] = (
    results.groupby("driverId")["positionOrder"]
    .shift(1)
    .rolling(5)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
)
 
# Elo ratings
INITIAL_RATING = 1500
K              = 20
SEASON_DECAY   = 0.85

def compute_elo(df, entity_col):
    """
    Compute hidden Elo ratings for drivers or constructors.
    Ratings decay toward the mean between seasons to account
    for team/driver changes.
    """
    elo      = defaultdict(lambda: INITIAL_RATING)
    out_idx  = []
    out_vals = []
    prev_year = None
 
    for race_id, race in df.groupby("raceId", sort=True):
        year = race["year"].iloc[0]
 
        if prev_year and year != prev_year:
            for k in elo:
                elo[k] = SEASON_DECAY * elo[k] + (1 - SEASON_DECAY) * INITIAL_RATING
        prev_year = year
 
        participants = race[[entity_col, "positionOrder"]].dropna().values
        n = len(participants)
        if n < 2:
            continue
 
        for (e1, p1), (e2, p2) in combinations(participants, 2):
            R1, R2 = elo[e1], elo[e2]
            E1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
            S1 = (n - p1) / (n - 1)
            elo[e1] += K * (S1 - E1)
            elo[e2] += K * ((1 - S1) - (1 - E1))
 
        for idx, row in race.iterrows():
            out_idx.append(idx)
            out_vals.append(elo.get(row[entity_col], INITIAL_RATING))
 
    return pd.Series(out_vals, index=out_idx)

results["driver_elo"]      = compute_elo(results, "driverId")
results["constructor_elo"] = compute_elo(results, "constructorId")

# ── Target Variables ──────────────────────────────────────────────────────────
 
results["points_finish"] = (results["points"] > 0).astype(int)

# ── Feature Selection ─────────────────────────────────────────────────────────
 
numerical_cols = [
    "grid", "round", "q1", "q2", "q3", "best_qualifying", "driver_age",
    "driver_points_ewma", "driver_last_finish", "driver_last_points",
    "driver_finish_trend_5", "constructor_points_rate_5",
    "driver_elo", "constructor_elo"
]
categorical_cols = ["constructorRef", "circuit_name"]
feature_cols = numerical_cols + categorical_cols
 
results[numerical_cols]   = results[numerical_cols].replace({pd.NA: np.nan}).infer_objects(copy=False)
results[categorical_cols] = results[categorical_cols].replace({pd.NA: np.nan}).infer_objects(copy=False)

# ── Train / Test Split ────────────────────────────────────────────────────────
 
train_mask = results["year"] <= TRAIN_CUTOFF_YEAR
test_mask  = results["year"] >  TRAIN_CUTOFF_YEAR
 
X_train = results.loc[train_mask, feature_cols]
y_train = results.loc[train_mask, "points_finish"]
X_test  = results.loc[test_mask,  feature_cols]
y_test  = results.loc[test_mask,  "points_finish"]
 
print(f"Train samples : {len(X_train):,}  ({results.loc[train_mask,'year'].min()}–{TRAIN_CUTOFF_YEAR})")
print(f"Test  samples : {len(X_test):,}  ({TRAIN_CUTOFF_YEAR+1}–{results.loc[test_mask,'year'].max()})")

# ── Pipeline ──────────────────────────────────────────────────────────────────
 
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])
 
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE))
])

# ── Train ─────────────────────────────────────────────────────────────────────
 
print("\nTraining logistic regression baseline...")
pipeline.fit(X_train, y_train)
 
# ── Evaluate ──────────────────────────────────────────────────────────────────
 
preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]
 
print("\n" + "="*60)
print("BASELINE MODEL RESULTS (Logistic Regression)")
print("="*60)
print(classification_report(y_test, preds))
print(f"ROC-AUC : {roc_auc_score(y_test, probs):.4f}")
 
# Threshold analysis
thresholds = np.linspace(0.05, 0.95, 50)
metrics_df = pd.DataFrame([
    {
        "threshold": t,
        "precision": precision_score(y_test, (probs >= t).astype(int), zero_division=0),
        "recall":    recall_score(y_test,    (probs >= t).astype(int), zero_division=0),
        "f1":        f1_score(y_test,        (probs >= t).astype(int), zero_division=0),
    }
    for t in thresholds
])
best_threshold = metrics_df.loc[metrics_df["f1"].idxmax(), "threshold"]
print(f"Best F1 threshold : {best_threshold:.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, probs):.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC Curve — Logistic Regression")
axes[0].legend()
 
# Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
axes[1].plot(prob_pred, prob_true, marker="o")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set(xlabel="Predicted Probability", ylabel="Observed Frequency",
            title="Calibration Curve — Logistic Regression")
 
# Threshold Analysis
axes[2].plot(metrics_df["threshold"], metrics_df["precision"], label="Precision")
axes[2].plot(metrics_df["threshold"], metrics_df["recall"],    label="Recall")
axes[2].plot(metrics_df["threshold"], metrics_df["f1"],        label="F1")
axes[2].axvline(best_threshold, color="red", linestyle="--", label=f"Best threshold ({best_threshold:.2f})")
axes[2].set(xlabel="Threshold", ylabel="Score", title="Threshold Analysis")
axes[2].legend()
 
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "baseline_evaluation.png", dpi=150)
plt.show()
print(f"\nPlots saved to {OUTPUT_PATH}baseline_evaluation.png")

# ── Save Predictions ──────────────────────────────────────────────────────────
 
output = results.loc[test_mask, [
    "year", "round", "circuit_name", "driverRef",
    "constructorRef", "positionOrder", "points"
]].copy()
output["actual_points_finish"] = y_test.values
output["pred_points_finish"]   = preds
output["prob_points_finish"]   = probs
output = output.sort_values(by=["year", "round", "positionOrder"])
output.to_csv(OUTPUT_PATH + "baseline_predictions.csv", index=False)
print(f"Predictions saved to {OUTPUT_PATH}baseline_predictions.csv")