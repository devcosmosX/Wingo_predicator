import sqlite3
import pandas as pd

def check_bias():
    try:
        conn = sqlite3.connect('wingo.db')
        df = pd.read_sql('SELECT digit FROM results', conn)
        conn.close()
        if df.empty:
            print("[!] No records found in wingo.db yet. Ensure scraper.py is running.")
            return
        print("=============================================")
        print("  WinGo 30S — Digit Distribution Analysis")
        print("=============================================")
        print(df['digit'].value_counts().sort_index())
        print(f"\nTotal records analyzed: {len(df)}")
    except Exception as e:
        print(f"[!] Error reading database: {e}")

if __name__ == "__main__":
    check_bias()
