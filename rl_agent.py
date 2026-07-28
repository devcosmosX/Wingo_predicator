import sqlite3, json
import numpy as np
import random

DB = "wingo.db"

class RLAgent:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.lr = 0.01
        self.gamma = 0.9
        self.history = []
        self._load_qtable()
    
    def _load_qtable(self):
        conn = sqlite3.connect(DB)
        rows = conn.execute("SELECT state_hash, action, q_value FROM rl_qtable").fetchall()
        self.q_table = {}
        for row in rows:
            key = (row[0], row[1])
            self.q_table[key] = row[2]
        conn.close()
        print(f"[RL] Loaded {len(self.q_table)} Q-values from DB")
    
    def _save_q(self, state_hash, action, value):
        conn = sqlite3.connect(DB)
        conn.execute(
            "INSERT OR REPLACE INTO rl_qtable (state_hash, action, q_value, visits) VALUES (?,?,?, COALESCE((SELECT visits FROM rl_qtable WHERE state_hash=? AND action=?),0)+1)",
            (state_hash, action, value, state_hash, action)
        )
        conn.commit()
        conn.close()
    
    def get_state_hash(self, last_5):
        return ",".join(str(d) for d in last_5)
    
    def predict(self, last_5, ml_pred=None, ml_conf=None):
        state = self.get_state_hash(last_5)
        
        # Get Q-values for this state
        q_vals = []
        for a in range(10):
            q_vals.append(self.q_table.get((state, a), 0.0))
        q_vals = np.array(q_vals)
        
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            chosen = random.randint(0, 9)
            mode = "explore"
        else:
            if ml_pred is not None and ml_conf is not None:
                ml_scores = np.ones(10) * 0.1
                ml_scores[ml_pred] = ml_conf
                # Softmax Q-values
                exp_q = np.exp(q_vals - q_vals.max()) if q_vals.max() > 0 else np.ones(10)/10
                rl_probs = exp_q / (exp_q.sum() + 1e-8)
                # Blend: 60% RL, 40% ML
                blended = 0.6 * rl_probs + 0.4 * ml_scores
                chosen = int(np.argmax(blended))
                mode = "blended"
            else:
                chosen = int(np.argmax(q_vals)) if q_vals.max() != 0 else random.randint(0, 9)
                mode = "q_only"
        
        return chosen, mode, state
    
    def learn(self, state_hash, action, actual_digit):
        reward = 1.0 if action == actual_digit else -0.5
        
        key = (state_hash, action)
        current_q = self.q_table.get(key, 0.0)
        
        # Max Q for this state
        max_q = max([self.q_table.get((state_hash, a), 0.0) for a in range(10)])
        
        new_q = current_q + self.lr * (reward + self.gamma * max_q - current_q)
        self.q_table[key] = new_q
        self._save_q(state_hash, action, new_q)
        
        self.history.append(action == actual_digit)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_accuracy(self, n=100):
        recent = self.history[-n:]
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_stats(self):
        return {
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "total_predictions": len(self.history),
            "accuracy": round(self.get_accuracy() * 100, 1)
        }
