"""
Training Pipeline v2.0
========================
Key fixes:
  - TimeSeriesSplit (forward-only, no data leakage) instead of shuffled StratifiedKFold
  - Trains 3 SEPARATE models: Digit (10-class), Size (binary), Color (3-class)
  - Feature importance analysis to prune noise features
  - Shared feature extraction function importable by server.py
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "wingo.db")


# ==================================================================
# Feature Extraction (shared between training and inference)
# ==================================================================

def extract_features(df):
    """
    Extracts all features from a DataFrame with columns: period, digit, color, size.
    Returns (df_with_features, feature_column_names).

    This function is importable by server.py so that training and inference
    use EXACTLY the same feature engineering pipeline.
    """
    df = df.copy()
    df["period_str"] = df["period"].astype(str)

    # --- Period ID features ---
    df["period_last_1"] = df["period_str"].str[-1:].apply(lambda x: int(x) if x.isdigit() else 0)
    df["period_last_2"] = df["period_str"].str[-2:].apply(lambda x: int(x) if x.isdigit() else 0)
    df["period_last_3"] = df["period_str"].str[-3:].apply(lambda x: int(x) if x.isdigit() else 0)
    df["period_last_3_mod10"] = df["period_last_3"] % 10
    df["period_digit_sum"] = df["period_str"].apply(
        lambda x: sum(int(c) for c in x if c.isdigit())
    )
    df["period_digit_sum_mod10"] = df["period_digit_sum"] % 10

    # --- Lag features (previous 8 digits) ---
    for i in range(1, 9):
        df[f"digit_lag_{i}"] = df["digit"].shift(i).fillna(5)  # fill with median

    # --- Rolling statistics ---
    df["last_5_mean"] = df["digit"].rolling(5, min_periods=1).mean()
    df["last_5_std"] = df["digit"].rolling(5, min_periods=1).std().fillna(0)
    df["last_10_mean"] = df["digit"].rolling(10, min_periods=1).mean()
    df["last_10_std"] = df["digit"].rolling(10, min_periods=1).std().fillna(0)
    df["last_20_mean"] = df["digit"].rolling(20, min_periods=1).mean()
    df["last_5_median"] = df["digit"].rolling(5, min_periods=1).median()
    df["last_5_max"] = df["digit"].rolling(5, min_periods=1).max()
    df["last_5_min"] = df["digit"].rolling(5, min_periods=1).min()
    df["last_5_range"] = df["last_5_max"] - df["last_5_min"]

    # --- Digit frequency in rolling windows ---
    for d in range(10):
        is_d = (df["digit"] == d).astype(float)
        df[f"freq_{d}_20"] = is_d.rolling(20, min_periods=1).mean()
        df[f"freq_{d}_50"] = is_d.rolling(50, min_periods=1).mean()

    # --- Hot/Cold gaps (draws since each digit last appeared) ---
    n = len(df)
    gaps = np.zeros((n, 10))
    last_seen = {d: -1 for d in range(10)}
    for idx in range(n):
        digit = int(df.iloc[idx]["digit"])
        for d in range(10):
            gaps[idx, d] = (idx - last_seen[d]) if last_seen[d] != -1 else (idx + 1)
        last_seen[digit] = idx
    for d in range(10):
        df[f"gap_{d}"] = gaps[:, d]

    # --- Color & size ratios ---
    color_str = df["color"].astype(str).str.lower().str.strip()
    df["is_green"] = (color_str == "green").astype(float)
    df["is_red"] = (color_str == "red").astype(float)
    df["is_violet"] = (color_str == "violet").astype(float)
    df["green_ratio_20"] = df["is_green"].rolling(20, min_periods=1).mean()
    df["red_ratio_20"] = df["is_red"].rolling(20, min_periods=1).mean()
    df["violet_ratio_20"] = df["is_violet"].rolling(20, min_periods=1).mean()

    df["is_big"] = (df["digit"] >= 5).astype(float)
    df["big_ratio_20"] = df["is_big"].rolling(20, min_periods=1).mean()

    # --- Size streak length ---
    size_num = (df["digit"] >= 5).astype(int)
    is_diff = size_num != size_num.shift(1)
    group_id = is_diff.cumsum()
    df["size_streak"] = df.groupby(group_id).cumcount()

    # --- Diff features ---
    df["digit_diff_1"] = df["digit"].diff(1).fillna(0)
    df["digit_diff_2"] = df["digit"].diff(2).fillna(0)
    df["digit_abs_diff_1"] = df["digit_diff_1"].abs()

    # Collect feature columns (exclude raw/target columns)
    exclude_cols = {
        "period", "period_str", "digit", "color", "size", "premium",
        "fetched_at", "target_digit", "target_size", "target_color",
        "is_green", "is_red", "is_violet", "is_big", "period_dt",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    return df, feature_cols


# ==================================================================
# Training
# ==================================================================

def train_and_evaluate():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT period, digit, color, size, fetched_at FROM results ORDER BY fetched_at ASC",
        conn,
    )
    conn.close()

    print(f"Total raw historical records loaded: {len(df)}")
    if len(df) < 80:
        print("Need at least 80 historical records to train!")
        return

    # --- Build targets (next round) ---
    df["target_digit"] = df["digit"].shift(-1)
    df["target_size"] = (df["digit"].shift(-1) >= 5).astype(float)

    color_map = {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 2, 7: 1, 8: 2, 9: 1}
    # 0=Violet, 1=Green, 2=Red
    df["target_color"] = df["digit"].shift(-1).map(color_map)

    # Extract features
    df, feature_cols = extract_features(df)

    # Drop last row (target unknown)
    df_clean = df.dropna(subset=["target_digit"]).copy()
    df_clean["target_digit"] = df_clean["target_digit"].astype(int)
    df_clean["target_size"] = df_clean["target_size"].astype(int)
    df_clean["target_color"] = df_clean["target_color"].astype(int)

    X = df_clean[feature_cols].values.astype(np.float32)
    y_digit = df_clean["target_digit"].values
    y_size = df_clean["target_size"].values
    y_color = df_clean["target_color"].values

    print(f"Dataset: X={X.shape}, Features={len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}... ({len(feature_cols)} total)")

    # --- TimeSeriesSplit (forward-only, no leakage) ---
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n--- DIGIT Model (10-class) ---")
    digit_scores = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        clf = LGBMClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        clf.fit(X[tr], y_digit[tr])
        acc = accuracy_score(y_digit[te], clf.predict(X[te]))
        digit_scores.append(acc)
        print(f"  Fold {fold + 1}: {acc * 100:.1f}%")
    print(f"  CV Digit Accuracy: {np.mean(digit_scores) * 100:.1f}%")

    print("\n--- SIZE Model (binary) ---")
    size_scores = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        clf = LGBMClassifier(
            n_estimators=150, learning_rate=0.04, max_depth=5,
            num_leaves=24, random_state=42, verbose=-1,
        )
        clf.fit(X[tr], y_size[tr])
        acc = accuracy_score(y_size[te], clf.predict(X[te]))
        size_scores.append(acc)
        print(f"  Fold {fold + 1}: {acc * 100:.1f}%")
    print(f"  CV Size Accuracy: {np.mean(size_scores) * 100:.1f}%")

    print("\n--- COLOR Model (3-class) ---")
    color_scores = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        clf = LGBMClassifier(
            n_estimators=150, learning_rate=0.04, max_depth=5,
            num_leaves=24, random_state=42, verbose=-1,
        )
        clf.fit(X[tr], y_color[tr])
        acc = accuracy_score(y_color[te], clf.predict(X[te]))
        color_scores.append(acc)
        print(f"  Fold {fold + 1}: {acc * 100:.1f}%")
    print(f"  CV Color Accuracy: {np.mean(color_scores) * 100:.1f}%")

    # --- Train final models on 100% data ---
    print("\nTraining final models on full dataset...")

    model_digit = LGBMClassifier(
        n_estimators=250, learning_rate=0.025, max_depth=6,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model_digit.fit(X, y_digit)

    model_size = LGBMClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        num_leaves=24, random_state=42, verbose=-1,
    )
    model_size.fit(X, y_size)

    model_color = LGBMClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        num_leaves=24, random_state=42, verbose=-1,
    )
    model_color.fit(X, y_color)

    # --- Feature importance ---
    print("\nTop 15 Digit Model Feature Importances:")
    imp = model_digit.feature_importances_
    sorted_idx = np.argsort(imp)[::-1]
    for i in range(min(15, len(feature_cols))):
        print(f"  {feature_cols[sorted_idx[i]]}: {imp[sorted_idx[i]]}")

    # --- Save ---
    joblib.dump(model_digit, os.path.join(BASE_DIR, "wingo_model_digit.pkl"))
    joblib.dump(model_size, os.path.join(BASE_DIR, "wingo_model_size.pkl"))
    joblib.dump(model_color, os.path.join(BASE_DIR, "wingo_model_color.pkl"))
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "wingo_features.pkl"))

    print(f"\nSaved 3 models + feature list ({len(feature_cols)} features)")
    print("  wingo_model_digit.pkl")
    print("  wingo_model_size.pkl")
    print("  wingo_model_color.pkl")
    print("  wingo_features.pkl")


if __name__ == "__main__":
    train_and_evaluate()
