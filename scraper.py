import requests, sqlite3, time, json
from datetime import datetime

DB = "wingo.db"
API = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

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

def init_db():
    conn = sqlite3.connect(DB)
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
            confidence REAL,
            mode TEXT,
            state_hash TEXT,
            actual_digit INTEGER,
            is_correct BOOLEAN,
            user_feedback BOOLEAN DEFAULT 0,
            predicted_at TEXT,
            resolved_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rl_qtable (
            state_hash TEXT,
            action INTEGER,
            q_value REAL DEFAULT 0.0,
            visits INTEGER DEFAULT 1,
            PRIMARY KEY (state_hash, action)
        )
    """)
    # Deduplicate existing records if any exist
    try:
        conn.execute("DELETE FROM predictions WHERE id NOT IN (SELECT MIN(id) FROM predictions GROUP BY period)")
        conn.execute("DELETE FROM results WHERE rowid NOT IN (SELECT MIN(rowid) FROM results GROUP BY period)")
    except: pass
    
    conn.commit()
    conn.close()
    print("[DB] Initialized & deduplicated SQLite database")

def fetch_results():
    ts = int(time.time() * 1000)
    try:
        r = requests.get(f"{API}?ts={ts}", headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("code") == 0 and data.get("data", {}).get("list"):
            return data["data"]["list"]
    except Exception as e:
        print(f"[API ERROR] {e}")
    return []

def store_results(draws):
    conn = sqlite3.connect(DB)
    inserted = 0
    now_str = datetime.now().isoformat()
    for d in draws:
        period_str = str(d["issueNumber"]).strip()
        digit = int(d["number"])
        size = "Big" if digit >= 5 else "Small"
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO results (period, digit, color, size, premium, fetched_at) VALUES (?,?,?,?,?,?)",
                (period_str, digit, d["color"], size, d.get("premium",""), now_str)
            )
            if cur.rowcount > 0:
                inserted += 1
        except: pass
    conn.commit()
    conn.close()
    return inserted

if __name__ == "__main__":
    init_db()
    print("[SCRAPER] Started — polling every 3 seconds with strict deduplication")
    while True:
        draws = fetch_results()
        if draws:
            n = store_results(draws)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(draws)} draws, {n} new unique")
        time.sleep(3)
