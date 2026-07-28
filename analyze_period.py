import sqlite3
from collections import Counter

def analyze_patterns():
    try:
        conn = sqlite3.connect('wingo.db')
        rows = conn.execute("SELECT period, digit FROM results ORDER BY fetched_at DESC LIMIT 500").fetchall()
        conn.close()
        
        if not rows:
            print("[!] No records found in wingo.db yet. Keep the scraper running.")
            return

        print("=============================================")
        print("   PERIOD ID & PATTERN ANALYSIS (500 Draws)")
        print("=============================================")
        print(f"Total draws analyzed: {len(rows)}\n")

        # Test 1: Does last digit of period match result?
        match_last = sum(1 for p,d in rows if str(p)[-1].isdigit() and int(str(p)[-1]) == d)
        print(f"1. Last digit of period ID = result digit: {match_last}/{len(rows)} ({match_last/len(rows)*100:.1f}%)")

        # Test 2: Does last 3 digits mod 10 match?
        match_mod = sum(1 for p,d in rows if str(p)[-3:].isdigit() and int(str(p)[-3:]) % 10 == d)
        print(f"2. Last 3 digits % 10 = result: {match_mod}/{len(rows)} ({match_mod/len(rows)*100:.1f}%)")

        # Test 3: Sum of all period digits % 10
        match_sum = sum(1 for p,d in rows if sum(int(c) for c in str(p) if c.isdigit()) % 10 == d)
        print(f"3. Sum of all digits % 10 = result: {match_sum}/{len(rows)} ({match_sum/len(rows)*100:.1f}%)")

        # Test 4: Check digit distribution
        dist = Counter(d for _,d in rows)
        print(f"\n4. Digit distribution (last {len(rows)}):")
        for i in range(10):
            bar = "#" * (dist[i] // 2)
            print(f"   Digit {i}: {dist[i]:3d} {bar}")

        # Test 5: Look for 30-second cycle patterns
        print(f"\n5. Cycle pattern detection (Same digit after N rounds):")
        for step in [5, 10, 15, 20, 30, 50]:
            if len(rows) > step:
                matches = sum(1 for i in range(len(rows) - step) if rows[i][1] == rows[i+step][1])
                total = len(rows) - step
                print(f"   Same digit after {step:2d} rounds: {matches}/{total} ({matches/total*100:.1f}%)")

        print("\n=============================================")
        print("Analysis complete!")

    except Exception as e:
        print(f"[!] Analysis error: {e}")

if __name__ == "__main__":
    analyze_patterns()
