import sqlite3, json, time, asyncio, os, sys, math
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import numpy as np
import pandas as pd

# ─── Vercel Serverless File Path Handling ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = os.environ.get("VERCEL") is not None

if IS_VERCEL or not os.access(BASE_DIR, os.W_OK):
    DB = "/tmp/wingo.db"
else:
    DB = os.path.join(BASE_DIR, "wingo.db")

API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-GB,en;q=0.7',
    'origin': 'https://www.tirangagame.xyz',
    'priority': 'u=1, i',
    'referer': 'https://www.tirangagame.xyz/',
    'sec-ch-ua': '"Not;A=Brand";v="8", "Chromium";v="150", "Brave";v="150"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36'
}

# ─── Load ML Model & RL Agent ───
model = None
features_list = None
rl_agent = None

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
    
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    cols_to_add = {
        "predicted_size": "TEXT",
        "predicted_color": "TEXT",
        "is_size_correct": "BOOLEAN",
        "is_color_correct": "BOOLEAN"
    }
    for col_name, col_type in cols_to_add.items():
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"[DB MIGRATION NOTE] {e}")

    conn.commit()
    conn.close()

init_db()

def load_ml():
    global model, features_list, rl_agent
    try:
        from autogluon.tabular import TabularPredictor
        model_path = os.path.join(BASE_DIR, 'wingo_model')
        if os.path.exists(model_path):
            model = TabularPredictor.load(model_path)
            print("[ML] AutoGluon model loaded successfully")
    except Exception:
        pass
    
    if model is None:
        try:
            import joblib
            model_pkl = os.path.join(BASE_DIR, 'wingo_model.pkl')
            feat_pkl = os.path.join(BASE_DIR, 'wingo_features.pkl')
            if os.path.exists(model_pkl):
                model = joblib.load(model_pkl)
                if os.path.exists(feat_pkl):
                    features_list = joblib.load(feat_pkl)
                print(f"[ML] Advanced LightGBM model loaded successfully ({len(features_list) if features_list else 0} features)")
        except Exception:
            try:
                import pickle
                model_pkl = os.path.join(BASE_DIR, 'wingo_model.pkl')
                if os.path.exists(model_pkl):
                    with open(model_pkl, 'rb') as f:
                        model = pickle.load(f)
                    print("[ML] LightGBM model (pickle) loaded successfully")
            except Exception as e2:
                print(f"[ML Note] {e2}")
    
    if model is None:
        print("[ML] No trained ML model found. Starting in RL-only mode.")
    
    try:
        from rl_agent import RLAgent
        rl_agent = RLAgent()
    except Exception as e:
        print(f"[RL ERROR] {e}")
        rl_agent = None

load_ml()

def get_color_for_digit(digit):
    if digit in (0, 5): return "Violet"
    elif digit in (1, 3, 7, 9): return "Green"
    else: return "Red"

def get_size_for_digit(digit):
    return "Big" if digit >= 5 else "Small"

# ─── WebSocket Connections & Cache ───
connected_clients = set()
current_pending_pred = None

def fetch_api():
    ts = int(time.time() * 1000)
    try:
        r = requests.get(f"{API_URL}?ts={ts}", headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("code") == 0 and data.get("data", {}).get("list"):
            return data["data"]["list"]
    except: pass
    return []

def sync_latest_draws_on_demand():
    """Fetches latest draws on demand for serverless requests"""
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
                    (period_str, dig, d["color"], size, now_str)
                )
            except: pass
        conn.commit()
        conn.close()
    return draws

def build_advanced_features_row(period_str):
    clean_period = str(period_str).strip()
    conn = get_db()
    df = pd.read_sql("SELECT period, digit, color, size, fetched_at FROM results ORDER BY fetched_at ASC", conn)
    conn.close()
    
    if len(df) < 5:
        return None
        
    df['period_str'] = df['period'].astype(str)
    df['period_last_1'] = df['period_str'].str[-1].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_2'] = df['period_str'].str[-2:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_3'] = df['period_str'].str[-3:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_last_3_mod10'] = df['period_last_3'] % 10
    df['period_last_4'] = df['period_str'].str[-4:].apply(lambda x: int(x) if x.isdigit() else 0)
    df['period_digit_sum'] = df['period_str'].apply(lambda x: sum(int(c) for c in x if c.isdigit()))
    df['period_digit_sum_mod10'] = df['period_digit_sum'] % 10

    for d in range(10):
        is_d = (df['digit'] == d).astype(int)
        df[f'freq_{d}_50'] = is_d.rolling(50, min_periods=1).mean().fillna(0.1)
        df[f'freq_{d}_20'] = is_d.rolling(20, min_periods=1).mean().fillna(0.1)

    gaps = np.zeros((len(df), 10))
    last_seen = {d: -1 for d in range(10)}
    for idx, digit in enumerate(df['digit'].values):
        for d in range(10):
            gaps[idx, d] = (idx - last_seen[d]) if last_seen[d] != -1 else idx
        last_seen[digit] = idx
    for d in range(10):
        df[f'gap_{d}'] = gaps[:, d]

    is_green = (df['color'] == 'green').astype(int)
    is_red = (df['color'] == 'red').astype(int)
    is_violet = (df['color'] == 'violet').astype(int)

    df['green_ratio_20'] = is_green.rolling(20, min_periods=1).mean().fillna(0.33)
    df['red_ratio_20'] = is_red.rolling(20, min_periods=1).mean().fillna(0.33)
    df['violet_ratio_20'] = is_violet.rolling(20, min_periods=1).mean().fillna(0.33)

    df['size_num'] = (df['digit'] >= 5).astype(int)
    is_diff = df['size_num'] != df['size_num'].shift(1)
    group_id = is_diff.cumsum()
    df['streak'] = df.groupby(group_id).cumcount().fillna(0)

    df['period_dt'] = pd.to_datetime(df['period_str'].str[:14], format='%Y%m%d%H%M%S', errors='coerce')
    df['hour'] = df['period_dt'].dt.hour.fillna(12).astype(int)
    df['minute'] = df['period_dt'].dt.minute.fillna(0).astype(int)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['round_of_day'] = df['period_last_4']

    for i in range(1, 7):
        df[f'digit_lag_{i}'] = df['digit'].shift(i-1).fillna(0)

    df['last_5_mean'] = df['digit'].rolling(5).mean().fillna(4.5)
    df['last_5_std'] = df['digit'].rolling(5).std().fillna(1.0)
    df['last_5_unique'] = df['digit'].rolling(5).apply(lambda x: len(set(x)), raw=True).fillna(3)

    last_row = df.iloc[-1].to_dict()
    if clean_period.isdigit():
        p_val = int(clean_period)
        last_row['period_last_1'] = p_val % 10
        last_row['period_last_2'] = p_val % 100
        last_row['period_last_3'] = p_val % 1000
        last_row['period_last_3_mod10'] = (p_val % 1000) % 10
        last_row['period_digit_sum_mod10'] = sum(int(c) for c in clean_period) % 10

    return last_row

def predict_next(last_5_digits, period_str):
    clean_period = str(period_str).strip()
    conn = get_db()
    
    existing = conn.execute(
        "SELECT rl_pred, confidence, mode, predicted_size, predicted_color FROM predictions WHERE period=? LIMIT 1",
        (clean_period,)
    ).fetchone()
    
    if existing:
        conn.close()
        pred_d = int(existing["rl_pred"])
        p_size = existing["predicted_size"] or get_size_for_digit(pred_d)
        p_color = existing["predicted_color"] or get_color_for_digit(pred_d)
        return pred_d, p_size, p_color, float(existing["confidence"]), existing["mode"]

    ml_pred = None
    ml_conf = None

    try:
        if model is not None:
            feat_row = build_advanced_features_row(clean_period)
            if feat_row and features_list:
                X_feat = np.array([[feat_row.get(f, 0.0) for f in features_list]])
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_feat)[0]
                    ml_pred = int(np.argmax(probs))
                    ml_conf = float(np.max(probs))
            elif hasattr(model, 'predict_proba'):
                feat = np.array([[last_5_digits[4], last_5_digits[3], last_5_digits[2],
                                last_5_digits[1], last_5_digits[0],
                                np.mean(last_5_digits), np.std(last_5_digits),
                                int(last_5_digits[4] == last_5_digits[3]),
                                sum(1 for d in last_5_digits if d >= 5)]])
                probs = model.predict_proba(feat)[0]
                ml_pred = int(np.argmax(probs))
                ml_conf = float(np.max(probs))
        
        if rl_agent is not None:
            rl_pred, mode, state_hash = rl_agent.predict(last_5_digits, ml_pred, ml_conf)
        else:
            rl_pred = ml_pred if ml_pred is not None else 0
            mode = "ml_only"
            state_hash = ",".join(str(d) for d in last_5_digits)
            ml_conf = ml_conf or 0.5

        pred_size = get_size_for_digit(rl_pred)
        pred_color = get_color_for_digit(rl_pred)

        now_str = datetime.now().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO predictions (period, ml_pred, rl_pred, predicted_size, predicted_color, confidence, mode, state_hash, predicted_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (clean_period, int(ml_pred) if ml_pred is not None else None, int(rl_pred), pred_size, pred_color, float(ml_conf if ml_conf is not None else 0.0), mode, state_hash, now_str)
        )
        conn.commit()
        conn.close()
        
        return int(rl_pred), pred_size, pred_color, float(ml_conf if ml_conf is not None else 0.0), mode
    except Exception as e:
        print(f"[PRED ERROR] {e}")
        conn.close()
        return None, "Small", "Green", 0, "error"

def compute_accuracies():
    conn = get_db()
    total = conn.execute("SELECT COUNT(DISTINCT period) FROM predictions WHERE actual_digit IS NOT NULL").fetchone()[0]
    digit_correct = conn.execute("SELECT COUNT(DISTINCT period) FROM predictions WHERE is_correct=1").fetchone()[0]
    
    try:
        size_correct = conn.execute("SELECT COUNT(DISTINCT period) FROM predictions WHERE is_size_correct=1").fetchone()[0]
    except: size_correct = 0
    try:
        color_correct = conn.execute("SELECT COUNT(DISTINCT period) FROM predictions WHERE is_color_correct=1").fetchone()[0]
    except: color_correct = 0

    conn.close()
    
    digit_acc = round(digit_correct / total * 100, 1) if total > 0 else 0.0
    size_acc = round(size_correct / total * 100, 1) if total > 0 else 0.0
    color_acc = round(color_correct / total * 100, 1) if total > 0 else 0.0
    
    return {
        "total": total,
        "digit_correct": digit_correct, "digit_acc": digit_acc,
        "size_correct": size_correct, "size_acc": size_acc,
        "color_correct": color_correct, "color_acc": color_acc
    }

async def background_scraper():
    global current_pending_pred
    last_processed_period = None

    while True:
        try:
            draws = sync_latest_draws_on_demand()
            if draws:
                latest = draws[0]
                period = str(latest["issueNumber"]).strip()
                digit = int(latest["number"])
                actual_size = "Big" if digit >= 5 else "Small"
                actual_color = str(latest["color"]).strip().capitalize()
                
                if period != last_processed_period:
                    last_processed_period = period
                    conn = get_db()
                    now_str = datetime.now().isoformat()
                    row = conn.execute(
                        "SELECT id, rl_pred, predicted_size, predicted_color, state_hash FROM predictions WHERE period=? AND actual_digit IS NULL ORDER BY id DESC LIMIT 1",
                        (period,)
                    ).fetchone()
                    
                    is_digit_correct = None
                    is_size_correct = None
                    is_color_correct = None
                    rl_pred_val = None
                    pred_size_val = None
                    pred_color_val = None
                    
                    if row:
                        pred_id, rl_pred_val, pred_size_val, pred_color_val, state_hash = row[0], row[1], row[2], row[3], row[4]
                        
                        if pred_size_val is None and rl_pred_val is not None: pred_size_val = get_size_for_digit(rl_pred_val)
                        if pred_color_val is None and rl_pred_val is not None: pred_color_val = get_color_for_digit(rl_pred_val)

                        is_digit_correct = (rl_pred_val == digit) if rl_pred_val is not None else False
                        is_size_correct = (pred_size_val.lower() == actual_size.lower()) if pred_size_val else False
                        
                        if pred_color_val:
                            if pred_color_val.lower() == actual_color.lower():
                                is_color_correct = True
                            elif actual_color.lower() in ('violet', 'red', 'green') and pred_color_val.lower() in actual_color.lower():
                                is_color_correct = True
                            else:
                                is_color_correct = False
                        else:
                            is_color_correct = False

                        try:
                            conn.execute(
                                "UPDATE predictions SET actual_digit=?, is_correct=?, is_size_correct=?, is_color_correct=?, predicted_size=?, predicted_color=?, resolved_at=? WHERE id=?",
                                (digit, is_digit_correct, is_size_correct, is_color_correct, pred_size_val, pred_color_val, now_str, pred_id)
                            )
                            conn.commit()
                        except Exception as e:
                            print(f"[UPDATE PRED ERROR] {e}")
                        
                        if rl_agent and state_hash and rl_pred_val is not None:
                            rl_agent.learn(state_hash, rl_pred_val, digit)
                    conn.close()

                    accs = compute_accuracies()

                    msg_result = json.dumps({
                        "type": "result",
                        "period": period,
                        "period_tail": str(period)[-3:],
                        "digit": digit,
                        "color": latest["color"],
                        "size": actual_size,
                        "predicted": rl_pred_val,
                        "predicted_size": pred_size_val,
                        "predicted_color": pred_color_val,
                        "correct_digit": is_digit_correct,
                        "correct_size": is_size_correct,
                        "correct_color": is_color_correct,
                        "digit_acc": accs["digit_acc"],
                        "size_acc": accs["size_acc"],
                        "color_acc": accs["color_acc"]
                    })
                    dead = set()
                    for ws in list(connected_clients):
                        try: await ws.send_text(msg_result)
                        except: dead.add(ws)
                    for ws in dead: connected_clients.discard(ws)

                conn = get_db()
                rows = conn.execute("SELECT digit FROM results ORDER BY fetched_at DESC LIMIT 5").fetchall()
                conn.close()
                
                if len(rows) >= 5:
                    last_5 = [r[0] for r in rows][::-1]
                    next_period = str(int(period) + 1)
                    pred_digit, pred_size, pred_color, conf, mode = predict_next(last_5, next_period)
                    accs = compute_accuracies()

                    if pred_digit is not None:
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
                            "epsilon": rl_agent.epsilon if rl_agent else 1.0
                        }
                        current_pending_pred = pred_payload
                        msg_pred = json.dumps(pred_payload)
                        dead = set()
                        for ws in list(connected_clients):
                            try: await ws.send_text(msg_pred)
                            except: dead.add(ws)
                        for ws in dead: connected_clients.discard(ws)

        except Exception as e:
            print(f"[BACKGROUND ERROR] {e}")
        
        await asyncio.sleep(3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not IS_VERCEL:
        task = asyncio.create_task(background_scraper())
        yield
        task.cancel()
    else:
        yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Routes ───
@app.get("/")
async def root():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/history")
async def history(page: int = 1, limit: int = 15):
    sync_latest_draws_on_demand()
    if page < 1: page = 1
    if limit < 1: limit = 15
    offset = (page - 1) * limit

    conn = get_db()
    total_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    total_pages = math.ceil(total_count / limit) if total_count > 0 else 1

    try:
        query = """
            SELECT 
                r.period, 
                r.digit, 
                r.color, 
                r.size, 
                r.fetched_at,
                p.rl_pred as predicted,
                p.predicted_size,
                p.predicted_color,
                p.is_correct as correct_digit,
                p.is_size_correct as correct_size,
                p.is_color_correct as correct_color
            FROM results r
            LEFT JOIN (
                SELECT period, rl_pred, predicted_size, predicted_color, is_correct, is_size_correct, is_color_correct,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY id DESC) as rn
                FROM predictions
            ) p ON r.period = p.period AND p.rn = 1
            ORDER BY r.fetched_at DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, (limit, offset)).fetchall()
    except Exception as e:
        print(f"[HISTORY SQL FALLBACK] {e}")
        query = """
            SELECT 
                r.period, 
                r.digit, 
                r.color, 
                r.size, 
                r.fetched_at,
                p.rl_pred as predicted,
                p.is_correct as correct_digit
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
            if not item.get("predicted_size"): item["predicted_size"] = get_size_for_digit(pred_d)
            if not item.get("predicted_color"): item["predicted_color"] = get_color_for_digit(pred_d)
            if item.get("correct_size") is None:
                item["correct_size"] = (item["predicted_size"].lower() == str(item["size"]).lower())
            if item.get("correct_color") is None:
                item["correct_color"] = (item["predicted_color"].lower() == str(item["color"]).lower())
        result_list.append(item)
        
    return {
        "page": page,
        "limit": limit,
        "total_records": total_count,
        "total_pages": total_pages,
        "data": result_list
    }

@app.get("/api/stats")
async def stats():
    sync_latest_draws_on_demand()
    try:
        accs = compute_accuracies()
        conn = get_db()
        total_results = conn.execute("SELECT COUNT(DISTINCT period) FROM results").fetchone()[0]
        conn.close()
        
        rl_stats = rl_agent.get_stats() if rl_agent else {}
        
        return {
            "total_results": total_results,
            "total_predictions": accs["total"],
            "digit_accuracy": accs["digit_acc"],
            "size_accuracy": accs["size_acc"],
            "color_accuracy": accs["color_acc"],
            "rl": rl_stats
        }
    except Exception as e:
        print(f"[STATS ERROR] {e}")
        return {
            "total_results": 0, "total_predictions": 0,
            "digit_accuracy": 0.0, "size_accuracy": 0.0, "color_accuracy": 0.0,
            "rl": {}
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
    pred_digit, pred_size, pred_color, conf, mode = predict_next(last_5, period_suffix)
    accs = compute_accuracies()
    
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
        "period_suffix": period_suffix
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
            "UPDATE predictions SET actual_digit=?, is_correct=?, is_size_correct=?, is_color_correct=?, resolved_at=?, user_feedback=1 WHERE period=?",
            (actual, predicted == actual, pred_size.lower() == actual_size.lower(), pred_color.lower() == actual_color.lower(), now_str, period)
        )
        conn.commit()
    except Exception as e:
        print(f"[FEEDBACK DB ERROR] {e}")
        
    if rl_agent:
        row = conn.execute(
            "SELECT state_hash, rl_pred FROM predictions WHERE period=? AND state_hash IS NOT NULL LIMIT 1",
            (period,)
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
        except: pass
        
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        connected_clients.discard(websocket)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    if len(sys.argv) > 1:
        try: port = int(sys.argv[1])
        except: pass
    uvicorn.run(app, host="0.0.0.0", port=port)
