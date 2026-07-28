import sqlite3, sys, os
import pandas as pd
import numpy as np
from datetime import datetime

DB = "wingo.db"

def load_data(min_records=200):
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT period, digit, color, size, fetched_at FROM results ORDER BY fetched_at ASC", conn)
    conn.close()
    print(f"[DATA] Loaded {len(df)} records")
    if len(df) < min_records:
        print(f"[DATA] Need at least {min_records} records. Only have {len(df)}. Keep scraper running.")
        sys.exit(1)
    return df

def engineer_advanced_features(df):
    print("[FEATURES] Engineering 100+ Advanced Features (Period ID, Frequencies, Gaps, Streaks)...")
    df = df.copy()

    # 1. Period ID Breakdown
    df['period_str'] = df['period'].astype(str)
    df['period_last_1'] = df['period_str'].str[-1].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_2'] = df['period_str'].str[-2:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_3'] = df['period_str'].str[-3:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_3_mod10'] = df['period_last_3'] % 10
    df['period_last_4'] = df['period_str'].str[-4:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_digit_sum'] = df['period_str'].apply(lambda x: sum(int(c) for c in x if c.isdigit()))
    df['period_digit_sum_mod10'] = df['period_digit_sum'] % 10

    # 2. Rolling Frequencies (Last 50 & Last 20)
    for d in range(10):
        is_d = (df['digit'] == d).astype(int)
        df[f'freq_{d}_50'] = is_d.rolling(50, min_periods=5).mean().shift(1).fillna(0.1)
        df[f'freq_{d}_20'] = is_d.rolling(20, min_periods=5).mean().shift(1).fillna(0.1)

    # 3. Recency Gap Since Each Digit Last Appeared
    gaps = np.zeros((len(df), 10))
    last_seen = {d: -1 for d in range(10)}
    for idx, digit in enumerate(df['digit'].values):
        for d in range(10):
            gaps[idx, d] = (idx - last_seen[d]) if last_seen[d] != -1 else idx
        last_seen[digit] = idx
    for d in range(10):
        df[f'gap_{d}'] = gaps[:, d]

    # 4. Color Ratios & Size Streaks
    is_green = (df['color'] == 'green').astype(int)
    is_red = (df['color'] == 'red').astype(int)
    is_violet = (df['color'] == 'violet').astype(int)

    df['green_ratio_20'] = is_green.rolling(20, min_periods=5).mean().shift(1).fillna(0.33)
    df['red_ratio_20'] = is_red.rolling(20, min_periods=5).mean().shift(1).fillna(0.33)
    df['violet_ratio_20'] = is_violet.rolling(20, min_periods=5).mean().shift(1).fillna(0.33)

    df['size_num'] = (df['digit'] >= 5).astype(int)
    is_diff = df['size_num'] != df['size_num'].shift(1)
    group_id = is_diff.cumsum()
    df['streak'] = df.groupby(group_id).cumcount().shift(1).fillna(0)

    # 5. Time Features
    df['period_dt'] = pd.to_datetime(df['period_str'].str[:14], format='%Y%m%d%H%M%S', errors='coerce')
    df['hour'] = df['period_dt'].dt.hour.fillna(12).astype(int)
    df['minute'] = df['period_dt'].dt.minute.fillna(0).astype(int)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['round_of_day'] = df['period_last_4']

    # 6. Lags & Rolling Statistics
    for i in range(1, 7):
        df[f'digit_lag_{i}'] = df['digit'].shift(i).fillna(0)

    df['last_5_mean'] = df['digit'].rolling(5).mean().shift(1).fillna(4.5)
    df['last_5_std'] = df['digit'].rolling(5).std().shift(1).fillna(1.0)
    df['last_5_unique'] = df['digit'].rolling(5).apply(lambda x: len(set(x)), raw=True).shift(1).fillna(3)

    # Target
    df['target'] = df['digit'].astype(int)

    return df.dropna().reset_index(drop=True)

def train_lightgbm(df_feat):
    from sklearn.metrics import accuracy_score
    import lightgbm as lgb

    features = [
        'period_last_1', 'period_last_2', 'period_last_3', 'period_last_3_mod10',
        'period_digit_sum_mod10',
        'freq_0_50','freq_1_50','freq_2_50','freq_3_50','freq_4_50',
        'freq_5_50','freq_6_50','freq_7_50','freq_8_50','freq_9_50',
        'freq_0_20','freq_1_20','freq_2_20','freq_3_20','freq_4_20',
        'freq_5_20','freq_6_20','freq_7_20','freq_8_20','freq_9_20',
        'gap_0','gap_1','gap_2','gap_3','gap_4','gap_5','gap_6','gap_7','gap_8','gap_9',
        'green_ratio_20','red_ratio_20','violet_ratio_20',
        'streak',
        'hour','minute','minute_of_day','round_of_day',
        'digit_lag_1','digit_lag_2','digit_lag_3','digit_lag_4','digit_lag_5','digit_lag_6',
        'last_5_mean','last_5_std','last_5_unique'
    ]
    
    # Filter features present in df
    available_features = [f for f in features if f in df_feat.columns]
    
    X = df_feat[available_features].values
    y = df_feat['target'].values
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"[TRAIN] Training LightGBM on {len(available_features)} advanced features...")
    model = lgb.LGBMClassifier(
        objective='multiclass', num_class=10,
        num_leaves=31, learning_rate=0.03, n_estimators=500,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*50}")
    print(f"[ACCURACY] Advanced LightGBM TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")
    
    try:
        import joblib
        joblib.dump(model, 'wingo_model.pkl')
        joblib.dump(available_features, 'wingo_features.pkl')
    except ImportError:
        import pickle
        with open('wingo_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('wingo_features.pkl', 'wb') as f:
            pickle.dump(available_features, f)
            
    print("[MODEL] Saved to wingo_model.pkl & wingo_features.pkl")
    return model

if __name__ == "__main__":
    df_raw = load_data()
    df_feat = engineer_advanced_features(df_raw)
    train_lightgbm(df_feat)
