import sqlite3
import pandas as pd
import numpy as np
import os, joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'wingo.db')

def extract_35_features(df):
    """
    Extracts 35+ advanced time-series, frequency, Markov, EWMA,
    and period ID modulo features for LightGBM training.
    """
    df = df.copy()
    df['period_str'] = df['period'].astype(str)
    
    # Target label: Next round's digit
    df['target'] = df['digit'].shift(-1)
    
    # 1. Period ID Modulo Features
    df['period_last_1'] = df['period_str'].str[-1].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_2'] = df['period_str'].str[-2:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_3'] = df['period_str'].str[-3:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_3_mod10'] = df['period_last_3'] % 10
    df['period_last_4'] = df['period_str'].str[-4:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_digit_sum'] = df['period_str'].apply(lambda x: sum(int(c) for c in x if c.isdigit()))
    df['period_digit_sum_mod10'] = df['period_digit_sum'] % 10

    # 2. Historical Frequency & Rolling Ratios (digits 0-9)
    for d in range(10):
        is_d = (df['digit'] == d).astype(float)
        df[f'freq_{d}_50'] = is_d.rolling(50, min_periods=1).mean()
        df[f'freq_{d}_20'] = is_d.rolling(20, min_periods=1).mean()

    # 3. Hot/Cold Digit Gaps (Draw distance since digit last appeared)
    gaps = np.zeros((len(df), 10))
    last_seen = {d: -1 for d in range(10)}
    for idx, digit in enumerate(df['digit'].values):
        for d in range(10):
            gaps[idx, d] = (idx - last_seen[d]) if last_seen[d] != -1 else idx
        last_seen[int(digit)] = idx
    for d in range(10):
        df[f'gap_{d}'] = gaps[:, d]

    # 4. Color & Size Streak Ratios
    is_green = (df['color'] == 'green').astype(float)
    is_red = (df['color'] == 'red').astype(float)
    is_violet = (df['color'] == 'violet').astype(float)
    
    df['green_ratio_20'] = is_green.rolling(20, min_periods=1).mean()
    df['red_ratio_20'] = is_red.rolling(20, min_periods=1).mean()
    df['violet_ratio_20'] = is_violet.rolling(20, min_periods=1).mean()

    df['size_num'] = (df['digit'] >= 5).astype(int)
    is_diff = df['size_num'] != df['size_num'].shift(1)
    group_id = is_diff.cumsum()
    df['streak'] = df.groupby(group_id).cumcount()

    # 5. Lag Features (Previous 6 digits)
    for i in range(1, 7):
        df[f'digit_lag_{i}'] = df['digit'].shift(i).fillna(0)

    # 6. Moving Averages & Volatility
    df['last_5_mean'] = df['digit'].rolling(5, min_periods=1).mean()
    df['last_5_std'] = df['digit'].rolling(5, min_periods=1).std().fillna(0)
    df['last_5_unique'] = df['digit'].rolling(5, min_periods=1).apply(lambda x: len(set(x)), raw=True)

    # Drop last row since target is unknown (next round)
    df_clean = df.dropna(subset=['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    
    feature_cols = [c for c in df_clean.columns if c not in ['period', 'period_str', 'digit', 'color', 'size', 'premium', 'fetched_at', 'target', 'period_dt']]
    return df_clean, feature_cols

def train_and_evaluate():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT period, digit, color, size, fetched_at FROM results ORDER BY fetched_at ASC", conn)
    conn.close()
    
    print(f"Total raw historical records loaded: {len(df)}")
    if len(df) < 50:
        print("Need at least 50 historical records to train ML model!")
        return

    df_clean, feature_cols = extract_35_features(df)
    X = df_clean[feature_cols].values
    y = df_clean['target'].values
    
    print(f"Dataset shape: X={X.shape}, Features={len(feature_cols)}")

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=5,
            num_leaves=24,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        print(f"Fold {fold+1} Digit Accuracy: {round(acc*100, 2)}%")

    print(f"5-Fold Cross-Validated Digit Accuracy: {round(np.mean(scores)*100, 2)}%")
    
    # Train final model on 100% of historical data
    final_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=24,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)

    model_out = os.path.join(BASE_DIR, 'wingo_model.pkl')
    feat_out = os.path.join(BASE_DIR, 'wingo_features.pkl')
    
    joblib.dump(final_model, model_out)
    joblib.dump(feature_cols, feat_out)
    print(f"Saved trained LightGBM model to {model_out}")
    print(f"Saved feature list ({len(feature_cols)} features) to {feat_out}")

if __name__ == "__main__":
    train_and_evaluate()
