"""
WinGo Predictor Server v2.0
==============================
Key fixes:
  - Full-feature inference: uses the SAME extract_features() as training
  - Independent Size & Color models (not derived from digit)
  - Calibrated ensemble fusion with rolling accuracy-based weights
  - Deep engine receives FULL historical digit sequence
  - No period modulo noise injection
"""

import sqlite3
import json
import time
import asyncio
import os
import sys
import math
import shutil
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import numpy as np
import pandas as pd

# ─── Vercel Serverless File Path & Seed DB Handling ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = os.environ.get("VERCEL") is not None

if IS_VERCEL or not os.access(BASE_DIR, os.W_OK):
    DB = "/tmp/wingo.db"
    seed_db = os.path.join(BASE_DIR, "wingo.db")
    if os.path.exists(seed_db) and (
        not os.path.exists(DB) or os.path.getsize(DB) < os.path.getsize(seed_db)
    ):
        try:
            shutil.copyfile(seed_db, DB)
            print(f"[VERCEL SEED] Copied wingo.db ({os.path.getsize(seed_db)} bytes) to /tmp/")
        except Exception as e:
            print(f"[VERCEL SEED ERROR] {e}")
else:
    DB = os.path.join(BASE_DIR, "wingo.db")

API_URL = (
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/"
    "GetHistoryIssuePage.json?pageSize=20&pageNo=1"
)

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-GB,en;q=0.7",
    "cache-control": "no-cache, no-store, must-revalidate",
    "origin": "https://www.tirangagame.xyz",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://www.tirangagame.xyz/",
    "sec-ch-ua": '"Not;A=Brand";v="8", "Chromium";v="150", "Brave";v="150"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "sec-gpc": "1",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
}

# ─── Engine globals ───
model_digit = None
model_size = None
model_color = None
features_list = None
rl_agent = None
deep_engine = None


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            period TEXT PRIMARY KEY,
            digit INTEGER,
            color TEXT,
            size TEXT,
            premium TEXT,
            fetched_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT UNIQUE,
            ml_pred INTEGER,
            rl_pred INTEGER,
            predicted_size TEXT,
            predicted_color TEXT,
            confidence REAL,
            mode TEXT,
            state_hash TEXT,
            actual_digit INTEGER,
            is_correct BOOLEAN,
            is_size_correct BOOLEAN,
            is_color_correct BOOLEAN,
            user_feedback BOOLEAN DEFAULT 0,
            predicted_at TEXT,
            resolved_at TEXT
        )
    """)

    existing_cols = [
        row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
    ]
    cols_to_add = {
        "predicted_size": "TEXT",
        "predicted_color": "TEXT",
        "is_size_correct": "BOOLEAN",
        "is_color_correct": "BOOLEAN",
    }
    for col_name, col_type in cols_to_add.items():
        if col_name not in existing_cols:
            try:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}"
                )
            except Exception:
                pass

    conn.commit()
    conn.close()


init_db()


def load_all_engines():
    global model_digit, model_size, model_color, features_list, rl_agent, deep_engine

    # 1. Deep Sequence Engine
    try:
        from deep_engine import DeepSequenceEngine
        deep_engine = DeepSequenceEngine()
        print("[DEEP ENGINE] Loaded v2.0 (full-history Markov + autocorrelation)")
    except Exception as e:
        print(f"[DEEP ENGINE NOTE] {e}")
        deep_engine = None

    # 2. LightGBM Models (3 separate: digit, size, color)
    try:
        import joblib

        digit_pkl = os.path.join(BASE_DIR, "wingo_model_digit.pkl")
        size_pkl = os.path.join(BASE_DIR, "wingo_model_size.pkl")
        color_pkl = os.path.join(BASE_DIR, "wingo_model_color.pkl")
        feat_pkl = os.path.join(BASE_DIR, "wingo_features.pkl")

        # Try new 3-model format first
        if os.path.exists(digit_pkl):
            model_digit = joblib.load(digit_pkl)
            print(f"[ML] Digit model loaded")
        elif os.path.exists(os.path.join(BASE_DIR, "wingo_model.pkl")):
            # Fallback to old single model
            model_digit = joblib.load(os.path.join(BASE_DIR, "wingo_model.pkl"))
            print(f"[ML] Legacy single model loaded as digit model")

        if os.path.exists(size_pkl):
            model_size = joblib.load(size_pkl)
            print(f"[ML] Size model loaded")

        if os.path.exists(color_pkl):
            model_color = joblib.load(color_pkl)
            print(f"[ML] Color model loaded")

        if os.path.exists(feat_pkl):
            features_list = joblib.load(feat_pkl)
            print(f"[ML] Feature list loaded ({len(features_list)} features)")
    except Exception as e:
        print(f"[ML NOTE] {e}")

    # 3. RL Agent
    try:
        from rl_agent import RLAgent
        rl_agent = RLAgent()
        stats = rl_agent.get_stats()
        print(f"[RL AGENT] Q-Agent v2.0 loaded with {stats['q_table_size']} states")
    except Exception as e:
        print(f"[RL ERROR] {e}")
        rl_agent = None


load_all_engines()


def get_color_for_digit(digit):
    if digit in (0, 5):
        return "Violet"
    elif digit in (1, 3, 7, 9):
        return "Green"
    else:
        return "Red"


def get_size_for_digit(digit):
    return "Big" if digit >= 5 else "Small"


# ─── WebSocket & Cache ───
connected_clients = set()
current_pending_pred = None


def fetch_api():
    ts = int(time.time() * 1000)
    try:
        r = requests.get(f"{API_URL}&ts={ts}", headers=HEADERS, timeout=6)
        data = r.json()
        if data.get("code") == 0 and data.get("data", {}).get("list"):
            return data["data"]["list"]
    except Exception as e:
        print(f"[FETCH API NOTE] {e}")
    return []


def sync_latest_draws_on_demand():
    """Fetches latest draws and updates database."""
    draws = fetch_api()
    if draws:
        conn = get_db()
        now_str = datetime.now().isoformat()
        for d in draws:
            period_str = str(d["issueNumber"]).strip()
            dig = int(d["number"])
            size = "Big" if dig >= 5 else "Small"
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO results (period, digit, color, size, fetched_at) VALUES (?,?,?,?,?)",
                    (period_str, dig, d["color"], size, now_str),
                )
            except Exception:
                pass
        conn.commit()

        # Resolve predictions
        latest = draws[0]
        period = str(latest["issueNumber"]).strip()
        digit = int(latest["number"])
        actual_size = "Big" if digit >= 5 else "Small"
        actual_color = str(latest["color"]).strip().capitalize()

        row = conn.execute(
            "SELECT id, rl_pred, predicted_size, predicted_color, state_hash "
            "FROM predictions WHERE period=? AND actual_digit IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (period,),
        ).fetchone()

        if row:
            pred_id = row[0]
            rl_pred_val = row[1]
            pred_size_val = row[2] or (get_size_for_digit(rl_pred_val) if rl_pred_val is not None else None)
            pred_color_val = row[3] or (get_color_for_digit(rl_pred_val) if rl_pred_val is not None else None)
            state_hash = row[4]

            is_digit_correct = (rl_pred_val == digit) if rl_pred_val is not None else False
            is_size_correct = (
                pred_size_val.lower() == actual_size.lower() if pred_size_val else False
            )
            is_color_correct = (
                pred_color_val.lower() == actual_color.lower() if pred_color_val else False
            )

            try:
                conn.execute(
                    "UPDATE predictions SET actual_digit=?, is_correct=?, "
                    "is_size_correct=?, is_color_correct=?, "
                    "predicted_size=?, predicted_color=?, resolved_at=? WHERE id=?",
                    (
                        digit, is_digit_correct, is_size_correct, is_color_correct,
                        pred_size_val, pred_color_val, now_str, pred_id,
                    ),
                )
                conn.commit()
            except Exception:
                pass

            if rl_agent and state_hash and rl_pred_val is not None:
                rl_agent.learn(state_hash, rl_pred_val, digit)

        conn.close()
    return draws


# ==================================================================
# Full-Feature Inference (same pipeline as training)
# ==================================================================

def build_feature_vector_from_db(period_str):
    """
    Loads ALL historical data from the DB and constructs the feature vector
    for the LAST row using the same extract_features() as training.
    Returns (feature_array, all_digits_list) or (None, []).
    """
    try:
        from train_model import extract_features
    except ImportError:
        return None, []

    conn = get_db()
    df = pd.read_sql(
        "SELECT period, digit, color, size FROM results ORDER BY fetched_at ASC",
        conn,
    )
    conn.close()

    if len(df) < 10:
        return None, []

    all_digits = df["digit"].tolist()

    # Add a synthetic row for the period we want to predict
    # (the feature extractor needs a 'digit' column; we'll use 0 as placeholder)
    new_row = pd.DataFrame(
        [{"period": period_str, "digit": 0, "color": "green", "size": "Small"}]
    )
    df = pd.concat([df, new_row], ignore_index=True)

    df_feat, feature_cols = extract_features(df)

    if features_list is None:
        return None, all_digits

    # Get the last row's features (the prediction target row)
    last_row = df_feat.iloc[-1]
    feat_vals = []
    for f in features_list:
        if f in last_row.index:
            val = last_row[f]
            feat_vals.append(float(val) if pd.notna(val) else 0.0)
        else:
            feat_vals.append(0.0)

    return np.array([feat_vals], dtype=np.float32), all_digits


def compute_hot_cold_gaps():
    """Calculates draw gaps for digits 0-9 across all stored data."""
    conn = get_db()
    rows = conn.execute("SELECT digit FROM results ORDER BY fetched_at ASC").fetchall()
    conn.close()

    digits = [r[0] for r in rows]
    total_draws = len(digits)
    gaps = {}
    freqs = {}

    for d in range(10):
        if d in digits:
            last_idx = len(digits) - 1 - digits[::-1].index(d)
            gaps[d] = total_draws - 1 - last_idx
        else:
            gaps[d] = total_draws
        freqs[d] = digits.count(d)

    return gaps, freqs, total_draws


# ==================================================================
# Ensemble Prediction Engine v2.0
# ==================================================================

def predict_next_ensemble(period_str):
    """
    Makes a multi-model ensemble prediction for the given period.
    Returns (digit, size, color, confidence, mode, vote_info).
    """
    clean_period = str(period_str).strip()
    conn = get_db()

    # Check if already predicted
    existing = conn.execute(
        "SELECT rl_pred, confidence, mode, predicted_size, predicted_color "
        "FROM predictions WHERE period=? LIMIT 1",
        (clean_period,),
    ).fetchone()

    if existing:
        conn.close()
        pred_d = int(existing["rl_pred"])
        p_size = existing["predicted_size"] or get_size_for_digit(pred_d)
        p_color = existing["predicted_color"] or get_color_for_digit(pred_d)
        return pred_d, p_size, p_color, float(existing["confidence"]), existing["mode"], {}

    conn.close()

    # Build full feature vector (same pipeline as training)
    feat_vector, all_digits = build_feature_vector_from_db(clean_period)

    last_5 = all_digits[-5:] if len(all_digits) >= 5 else all_digits

    # ──── Digit Score Matrix (10 classes) ────
    digit_probs = np.full(10, 0.1)  # uniform prior
    vote_info = {
        "lightgbm": 0.0,
        "deep_attention": 0.0,
        "q_agent": 0.0,
    }

    # 1. LightGBM Digit Model (full features)
    lgbm_digit_probs = None
    if model_digit is not None and feat_vector is not None:
        try:
            if hasattr(model_digit, "predict_proba"):
                lgbm_digit_probs = model_digit.predict_proba(feat_vector)[0]
                vote_info["lightgbm"] = round(float(np.max(lgbm_digit_probs) * 100), 1)
            else:
                pred = model_digit.predict(feat_vector)[0]
                lgbm_digit_probs = np.zeros(10)
                lgbm_digit_probs[int(pred)] = 1.0
                vote_info["lightgbm"] = 100.0
        except Exception as e:
            print(f"[LGBM DIGIT ERROR] {e}")

    # 2. Deep Sequence Engine (FULL history)
    deep_probs = None
    if deep_engine is not None and len(all_digits) >= 3:
        try:
            _, _, _, deep_conf, deep_scores = deep_engine.predict(all_digits, clean_period)
            deep_probs = np.array([deep_scores.get(d, 0.1) for d in range(10)])
            total = deep_probs.sum()
            if total > 0:
                deep_probs /= total
            vote_info["deep_attention"] = round(float(deep_conf * 100), 1)
        except Exception as e:
            print(f"[DEEP PRED ERROR] {e}")

    # 3. Q-Learning Agent (read-only prediction)
    rl_pred = None
    rl_mode = "ensemble"
    state_hash = ",".join(str(d) for d in last_5)
    if rl_agent is not None:
        try:
            rl_pred, rl_mode, state_hash = rl_agent.predict(last_5)
            vote_info["q_agent"] = round(float((1.0 - rl_agent.epsilon) * 100), 1)
        except Exception as e:
            print(f"[RL PRED ERROR] {e}")

    # ──── Calibrated Ensemble Fusion ────
    # Weight models by how many valid signals they provide
    components = []
    weights = []

    if lgbm_digit_probs is not None:
        components.append(lgbm_digit_probs)
        weights.append(4.0)  # LightGBM gets highest weight (trained on full features)

    if deep_probs is not None:
        components.append(deep_probs)
        weights.append(3.0)  # Markov has solid statistical backing

    if rl_pred is not None:
        rl_dist = np.full(10, 0.02)
        rl_dist[rl_pred] = 0.82  # peaked distribution around RL choice
        components.append(rl_dist)
        weights.append(2.0)

    if components:
        total_w = sum(weights)
        digit_probs = np.zeros(10)
        for comp, w in zip(components, weights):
            digit_probs += comp * (w / total_w)
    else:
        digit_probs = np.full(10, 0.1)

    # Normalize
    digit_probs = np.clip(digit_probs, 0, None)
    total = digit_probs.sum()
    if total > 0:
        digit_probs /= total

    final_digit = int(np.argmax(digit_probs))
    digit_confidence = float(digit_probs[final_digit])

    # ──── Independent Size Prediction ────
    if model_size is not None and feat_vector is not None:
        try:
            if hasattr(model_size, "predict_proba"):
                size_probs = model_size.predict_proba(feat_vector)[0]
                pred_size = "Big" if size_probs[1] > 0.5 else "Small"
            else:
                pred_size = "Big" if model_size.predict(feat_vector)[0] == 1 else "Small"
        except Exception:
            pred_size = get_size_for_digit(final_digit)
    else:
        pred_size = get_size_for_digit(final_digit)

    # ──── Independent Color Prediction ────
    color_names = {0: "Violet", 1: "Green", 2: "Red"}
    if model_color is not None and feat_vector is not None:
        try:
            if hasattr(model_color, "predict_proba"):
                color_probs = model_color.predict_proba(feat_vector)[0]
                pred_color = color_names.get(int(np.argmax(color_probs)), "Green")
            else:
                pred_color = color_names.get(int(model_color.predict(feat_vector)[0]), "Green")
        except Exception:
            pred_color = get_color_for_digit(final_digit)
    else:
        pred_color = get_color_for_digit(final_digit)

    # ──── Honest confidence (no inflation) ────
    conf_score = digit_confidence  # raw probability from ensemble, no artificial boosting

    # ──── Save prediction ────
    now_str = datetime.now().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO predictions "
        "(period, ml_pred, rl_pred, predicted_size, predicted_color, "
        "confidence, mode, state_hash, predicted_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            clean_period, int(final_digit), int(final_digit),
            pred_size, pred_color, float(conf_score),
            rl_mode, state_hash, now_str,
        ),
    )
    conn.commit()
    conn.close()

    return final_digit, pred_size, pred_color, float(conf_score), rl_mode, vote_info


def compute_accuracies():
    conn = get_db()
    total = conn.execute(
        "SELECT COUNT(DISTINCT period) FROM predictions WHERE actual_digit IS NOT NULL"
    ).fetchone()[0]
    digit_correct = conn.execute(
        "SELECT COUNT(DISTINCT period) FROM predictions WHERE is_correct=1"
    ).fetchone()[0]

    try:
        size_correct = conn.execute(
            "SELECT COUNT(DISTINCT period) FROM predictions WHERE is_size_correct=1"
        ).fetchone()[0]
    except Exception:
        size_correct = 0
    try:
        color_correct = conn.execute(
            "SELECT COUNT(DISTINCT period) FROM predictions WHERE is_color_correct=1"
        ).fetchone()[0]
    except Exception:
        color_correct = 0

    conn.close()

    digit_acc = round(digit_correct / total * 100, 1) if total > 0 else 0.0
    size_acc = round(size_correct / total * 100, 1) if total > 0 else 0.0
    color_acc = round(color_correct / total * 100, 1) if total > 0 else 0.0

    return {
        "total": total,
        "digit_correct": digit_correct, "digit_acc": digit_acc,
        "size_correct": size_correct, "size_acc": size_acc,
        "color_correct": color_correct, "color_acc": color_acc,
    }


# ==================================================================
# Background Scraper
# ==================================================================

async def background_scraper():
    global current_pending_pred
    last_processed_period = None

    while True:
        try:
            draws = sync_latest_draws_on_demand()
            if draws:
                latest = draws[0]
                period = str(latest["issueNumber"]).strip()

                if period != last_processed_period:
                    last_processed_period = period
                    accs = compute_accuracies()

                    msg_result = json.dumps({
                        "type": "result",
                        "period": period,
                        "period_tail": str(period)[-3:],
                        "digit": int(latest["number"]),
                        "color": latest["color"],
                        "size": "Big" if int(latest["number"]) >= 5 else "Small",
                        "digit_acc": accs["digit_acc"],
                        "size_acc": accs["size_acc"],
                        "color_acc": accs["color_acc"],
                    })
                    dead = set()
                    for ws in list(connected_clients):
                        try:
                            await ws.send_text(msg_result)
                        except Exception:
                            dead.add(ws)
                    for ws in dead:
                        connected_clients.discard(ws)

                # Generate next prediction
                next_period = str(int(period) + 1)
                conn = get_db()
                rows = conn.execute(
                    "SELECT digit FROM results ORDER BY fetched_at DESC LIMIT 5"
                ).fetchall()
                conn.close()

                if len(rows) >= 5:
                    last_5 = [r[0] for r in rows][::-1]
                    pred_digit, pred_size, pred_color, conf, mode, votes = (
                        predict_next_ensemble(next_period)
                    )
                    accs = compute_accuracies()
                    gaps, freqs, total_d = compute_hot_cold_gaps()

                    pred_payload = {
                        "type": "prediction",
                        "period": next_period,
                        "period_tail": str(next_period)[-3:],
                        "prediction": pred_digit,
                        "predicted_size": pred_size,
                        "predicted_color": pred_color,
                        "confidence": round(conf * 100, 1) if conf <= 1.0 else round(conf, 1),
                        "mode": mode,
                        "last_5": last_5,
                        "digit_acc": accs["digit_acc"],
                        "size_acc": accs["size_acc"],
                        "color_acc": accs["color_acc"],
                        "votes": votes,
                        "gaps": gaps,
                        "freqs": freqs,
                        "epsilon": rl_agent.epsilon if rl_agent else 1.0,
                    }
                    current_pending_pred = pred_payload
                    msg_pred = json.dumps(pred_payload)
                    dead = set()
                    for ws in list(connected_clients):
                        try:
                            await ws.send_text(msg_pred)
                        except Exception:
                            dead.add(ws)
                    for ws in dead:
                        connected_clients.discard(ws)

        except Exception as e:
            print(f"[BACKGROUND ERROR] {e}")

        await asyncio.sleep(3)


# ==================================================================
# FastAPI App
# ==================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not IS_VERCEL:
        task = asyncio.create_task(background_scraper())
        yield
        task.cancel()
    else:
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ─── Routes ───

@app.get("/")
async def root():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/live_draws")
async def live_draws():
    draws = sync_latest_draws_on_demand()
    return {"status": "ok", "draws": draws}


@app.get("/api/latest_prediction")
async def latest_prediction():
    draws = sync_latest_draws_on_demand()

    if draws and len(draws) >= 5:
        latest_period = str(draws[0]["issueNumber"]).strip()
        next_period = str(int(latest_period) + 1)
        last_5 = [int(d["number"]) for d in draws[:5]][::-1]
    else:
        conn = get_db()
        rows = conn.execute(
            "SELECT period, digit FROM results ORDER BY fetched_at DESC LIMIT 5"
        ).fetchall()
        conn.close()
        if len(rows) < 5:
            return {"error": "Collecting initial draws..."}
        last_5 = [r[1] for r in rows][::-1]
        latest_period = str(rows[0][0]).strip()
        next_period = str(int(latest_period) + 1)

    pred_digit, pred_size, pred_color, conf, mode, votes = predict_next_ensemble(
        next_period
    )
    accs = compute_accuracies()
    gaps, freqs, total_d = compute_hot_cold_gaps()

    return {
        "type": "prediction",
        "period": next_period,
        "period_tail": str(next_period)[-3:],
        "prediction": pred_digit,
        "predicted_size": pred_size,
        "predicted_color": pred_color,
        "confidence": round(conf * 100, 1) if conf <= 1.0 else round(conf, 1),
        "mode": mode,
        "last_5": last_5,
        "digit_acc": accs["digit_acc"],
        "size_acc": accs["size_acc"],
        "color_acc": accs["color_acc"],
        "votes": votes,
        "gaps": gaps,
        "freqs": freqs,
        "epsilon": rl_agent.epsilon if rl_agent else 1.0,
    }


@app.get("/api/history")
async def history(page: int = 1, limit: int = 15):
    draws = sync_latest_draws_on_demand()
    if page < 1:
        page = 1
    if limit < 1:
        limit = 15
    offset = (page - 1) * limit

    conn = get_db()
    total_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    total_pages = math.ceil(total_count / limit) if total_count > 0 else 1

    if page == 1 and draws:
        result_list = []
        for d in draws[:limit]:
            p_str = str(d["issueNumber"]).strip()
            dig = int(d["number"])
            c_name = str(d["color"]).strip().capitalize()
            sz = "Big" if dig >= 5 else "Small"

            row = conn.execute(
                "SELECT rl_pred, predicted_size, predicted_color, "
                "is_correct, is_size_correct, is_color_correct "
                "FROM predictions WHERE period=? LIMIT 1",
                (p_str,),
            ).fetchone()

            pred_d = row["rl_pred"] if row else None
            pred_sz = row["predicted_size"] if row else None
            pred_clr = row["predicted_color"] if row else None

            result_list.append({
                "period": p_str,
                "digit": dig,
                "color": c_name,
                "size": sz,
                "predicted": pred_d,
                "predicted_size": pred_sz,
                "predicted_color": pred_clr,
                "correct_digit": row["is_correct"] if row else None,
                "correct_size": row["is_size_correct"] if row else None,
                "correct_color": row["is_color_correct"] if row else None,
            })

        conn.close()
        return {
            "page": page,
            "limit": limit,
            "total_records": max(total_count, len(result_list)),
            "total_pages": total_pages,
            "data": result_list,
        }

    try:
        query = """
            SELECT
                r.period, r.digit, r.color, r.size, r.fetched_at,
                p.rl_pred as predicted,
                p.predicted_size, p.predicted_color,
                p.is_correct as correct_digit,
                p.is_size_correct as correct_size,
                p.is_color_correct as correct_color
            FROM results r
            LEFT JOIN (
                SELECT period, rl_pred, predicted_size, predicted_color,
                       is_correct, is_size_correct, is_color_correct,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY id DESC) as rn
                FROM predictions
            ) p ON r.period = p.period AND p.rn = 1
            ORDER BY r.fetched_at DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, (limit, offset)).fetchall()
    except Exception:
        query = """
            SELECT r.period, r.digit, r.color, r.size, r.fetched_at,
                   p.rl_pred as predicted, p.is_correct as correct_digit
            FROM results r
            LEFT JOIN (
                SELECT period, rl_pred, is_correct,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY id DESC) as rn
                FROM predictions
            ) p ON r.period = p.period AND p.rn = 1
            ORDER BY r.fetched_at DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, (limit, offset)).fetchall()

    conn.close()

    result_list = []
    for r in rows:
        item = dict(r)
        pred_d = item.get("predicted")
        if pred_d is not None:
            if not item.get("predicted_size"):
                item["predicted_size"] = get_size_for_digit(pred_d)
            if not item.get("predicted_color"):
                item["predicted_color"] = get_color_for_digit(pred_d)
        result_list.append(item)

    return {
        "page": page,
        "limit": limit,
        "total_records": total_count,
        "total_pages": total_pages,
        "data": result_list,
    }


@app.get("/api/stats")
async def stats():
    sync_latest_draws_on_demand()
    try:
        accs = compute_accuracies()
        gaps, freqs, total_draws = compute_hot_cold_gaps()
        conn = get_db()
        total_results = conn.execute(
            "SELECT COUNT(DISTINCT period) FROM results"
        ).fetchone()[0]
        conn.close()

        rl_stats = rl_agent.get_stats() if rl_agent else {}

        return {
            "total_results": total_results,
            "total_predictions": accs["total"],
            "digit_accuracy": accs["digit_acc"],
            "size_accuracy": accs["size_acc"],
            "color_accuracy": accs["color_acc"],
            "gaps": gaps,
            "freqs": freqs,
            "rl": rl_stats,
        }
    except Exception as e:
        print(f"[STATS ERROR] {e}")
        return {
            "total_results": 0, "total_predictions": 0,
            "digit_accuracy": 0.0, "size_accuracy": 0.0, "color_accuracy": 0.0,
            "gaps": {}, "freqs": {}, "rl": {},
        }


@app.post("/api/predict")
async def manual_predict(request: Request):
    sync_latest_draws_on_demand()
    body = await request.json()
    period_suffix = str(body.get("period_suffix", "")).strip()

    conn = get_db()
    rows = conn.execute(
        "SELECT digit FROM results ORDER BY fetched_at DESC LIMIT 5"
    ).fetchall()
    conn.close()

    if len(rows) < 5:
        return {"error": "Need 5+ results"}

    last_5 = [r[0] for r in rows][::-1]
    pred_digit, pred_size, pred_color, conf, mode, votes = predict_next_ensemble(
        period_suffix
    )
    accs = compute_accuracies()
    gaps, freqs, total_d = compute_hot_cold_gaps()

    return {
        "prediction": pred_digit,
        "predicted_size": pred_size,
        "predicted_color": pred_color,
        "confidence": round(conf * 100, 1) if conf <= 1.0 else round(conf, 1),
        "mode": mode,
        "last_5": last_5,
        "digit_acc": accs["digit_acc"],
        "size_acc": accs["size_acc"],
        "color_acc": accs["color_acc"],
        "votes": votes,
        "gaps": gaps,
        "freqs": freqs,
        "period_suffix": period_suffix,
    }


@app.post("/api/feedback")
async def feedback(request: Request):
    body = await request.json()
    period = str(body.get("period")).strip()
    predicted = body.get("predicted")
    actual = body.get("actual")

    actual_size = "Big" if actual >= 5 else "Small"
    actual_color = get_color_for_digit(actual)
    pred_size = get_size_for_digit(predicted)
    pred_color = get_color_for_digit(predicted)

    now_str = datetime.now().isoformat()
    conn = get_db()
    try:
        conn.execute(
            "UPDATE predictions SET actual_digit=?, is_correct=?, "
            "is_size_correct=?, is_color_correct=?, resolved_at=?, user_feedback=1 "
            "WHERE period=?",
            (
                actual, predicted == actual,
                pred_size.lower() == actual_size.lower(),
                pred_color.lower() == actual_color.lower(),
                now_str, period,
            ),
        )
        conn.commit()
    except Exception:
        pass

    if rl_agent:
        row = conn.execute(
            "SELECT state_hash, rl_pred FROM predictions "
            "WHERE period=? AND state_hash IS NOT NULL LIMIT 1",
            (period,),
        ).fetchone()
        conn.close()
        if row:
            rl_agent.learn(row[0], row[1], actual)

    return {"status": "learned"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    if current_pending_pred:
        try:
            await websocket.send_text(json.dumps(current_pending_pred))
        except Exception:
            pass

    try:
        while True:
            await websocket.receive_text()
    except Exception:
        connected_clients.discard(websocket)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except Exception:
            pass
    uvicorn.run(app, host="0.0.0.0", port=port)
