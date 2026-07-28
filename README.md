# ⚡ WinGo 30S Predictor

A real-time AI-powered lottery draw prediction system utilizing AutoML and Reinforcement Learning (Q-learning) with dynamic feedback loops.

---

## ⚡ Quick Start Guide (Windows & Cross-Platform)

### Step 1: Install Dependencies
Open PowerShell or Command Prompt inside the `wingo-predictor` directory:
```powershell
cd wingo-predictor
pip install -r requirements.txt
```
*(Optional for maximum accuracy)*:
```powershell
pip install autogluon.tabular
```

---

### Step 2: Start the Data Scraper (Terminal 1 — Keep Running)
```powershell
python scraper.py
```
*Leave this running! It fetches new draw results every 3 seconds and populates `wingo.db`.*

---

### Step 3: Train the Machine Learning Model (Terminal 2)
Open a **new** terminal tab/window:
```powershell
cd wingo-predictor
# Wait until the scraper has logged 200+ records (watch Terminal 1)
python train.py
```

---

### Step 4: Launch the Live Predictor Server (Terminal 3)
Open **another** terminal window:
```powershell
cd wingo-predictor
python server.py
# Or on PowerShell: .\run.ps1
# Or on Linux/Git Bash: bash run.sh
```

---

### Step 5: Open Dashboard
Open your browser and navigate to:  
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🧠 How It All Works

### The Data Flow Architecture

```
draw.ar-lottery01.com API
        │
        ▼
   scraper.py (polls every 3 sec)
        │
        ▼
   SQLite Database (wingo.db)
        │
        ├──▶ train.py (AutoML — identifies historical patterns & features)
        │         │
        │         ▼
        │    Saved Model (wingo_model/ or wingo_model.pkl)
        │
        └──▶ server.py (FastAPI + WebSocket Engine)
                  │
                  ├──▶ Loads ML model predictions
                  ├──▶ Runs RL Agent (Q-Learning blends ML & exploration)
                  ├──▶ Streams updates to Browser via WebSockets
                  └──▶ Accepts User Corrections via Feedback API
```

---

### 🔄 The RL Feedback Cycle

#### Success Scenario (Positive Reward):
1. **AutoML** predicts digit `7` (based on rolling statistics & lag patterns).
2. **RL Agent** blends Q-learning weights $\rightarrow$ decides on action `3`.
3. Actual result arrives: `3` ✅ **(Match!)**
   - RL receives **Reward +1.0**.
   - Q-value for state `(last_5_hash, action=3)` increases.

#### Correction Scenario (Negative Reward & Re-learning):
1. **AutoML** predicts digit `5`.
2. **RL Agent** predicts `5`.
3. Actual result arrives: `8` ❌ **(Mismatch!)**
   - Dashboard prompts user with the correction form.
   - User inputs or clicks `8`.
   - RL receives **Reward -0.5**.
   - Q-value for `(state, action=5)` decreases, while Q-value for `(state, action=8)` is reinforced.

---

## 📁 File Structure & Component Reference

| File / Folder | Purpose |
| :--- | :--- |
| [scraper.py](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/scraper.py) | Continuously polls the live draw API and stores results in `wingo.db`. |
| [rl_agent.py](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/rl_agent.py) | Q-learning agent that continuously learns from correct & incorrect predictions. |
| [train.py](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/train.py) | AutoML model training pipeline using AutoGluon with LightGBM fallback. |
| [server.py](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/server.py) | FastAPI application with real-time WebSockets, ML engine, and feedback endpoints. |
| [index.html](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/index.html) | Modern dark-mode UI with live digit animations, accuracy counters, and feedback buttons. |
| [check_patterns.py](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/check_patterns.py) | Diagnostic utility to test for digit bias and distribution anomalies in `wingo.db`. |
| [run.ps1](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/run.ps1) / [run.sh](file:///c:/Users/GhanshamDeepakGavand/OneDrive%20-%20IBM/Projects/BAX/wingo-predictor/run.sh) | Launcher scripts for Windows PowerShell and Bash environments. |

---

## 📊 Understanding Results & Statistical Insights

- **Initial Accuracy**: Starts near **10%** because the RL agent begins in exploration mode (`epsilon = 1.0`).
- **Learning Curve**: As `epsilon` decays (`decay = 0.998`), the system relies increasingly on blended Q-table values and AutoML predictions.
- **Pure PRNG vs. Biased Systems**:
  - If the RNG is uniform and unbiased, theoretical long-term accuracy caps around **10%** (1 in 10 uniform chance).
  - If draw mechanics exhibit streak biases, color imbalances, or temporal repetition, accuracy can reach **15%–30%+**.
- **Checking Digit Bias**:
  Run the included pattern check script:
  ```powershell
  python check_patterns.py
  ```
