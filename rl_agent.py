import numpy as np
import os, pickle

class RLAgent:
    """
    Q-Learning Reinforcement Learning Agent for WinGo 30S.
    State representation includes 3-gram digit sequence + size streak + color streak.
    Reward function awards +3.0 for exact digit match, +1.0 for size/color match, -1.0 for miss.
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.10, min_epsilon=0.01, decay=0.999):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q_table = {}
        self.load()

    def get_state_hash(self, last_5_digits, streak_size=0, streak_color=0):
        if len(last_5_digits) < 3:
            s_str = ",".join(str(d) for d in last_5_digits)
        else:
            s_str = f"{last_5_digits[-3]},{last_5_digits[-2]},{last_5_digits[-1]}|s{streak_size}|c{streak_color}"
        return s_str

    def predict(self, last_5_digits, ml_pred=None, ml_conf=None):
        state = self.get_state_hash(last_5_digits)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(10)

        q_values = self.q_table[state]

        if ml_pred is not None and 0 <= ml_pred <= 9:
            boost = (ml_conf or 0.8) * 1.5
            q_values[ml_pred] += boost

        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(10))
            mode = "rl_explore"
        else:
            action = int(np.argmax(q_values))
            mode = "rl_exploit"

        return action, mode, state

    def learn(self, state, action, actual_digit):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(10)

        # Asymmetric multi-target reward calculation
        if action == actual_digit:
            reward = 3.5 # Big hit for exact digit
        else:
            pred_size = "Big" if action >= 5 else "Small"
            actual_size = "Big" if actual_digit >= 5 else "Small"
            
            pred_color = "Violet" if action in (0,5) else ("Green" if action in (1,3,7,9) else "Red")
            actual_color = "Violet" if actual_digit in (0,5) else ("Green" if actual_digit in (1,3,7,9) else "Red")

            reward = 0.0
            if pred_size == actual_size: reward += 1.0
            else: reward -= 0.5
            if pred_color == actual_color: reward += 1.0
            else: reward -= 0.5

        old_val = self.q_table[state][action]
        max_future = np.max(self.q_table[state])
        
        # Q-learning update formula
        self.q_table[state][action] = old_val + self.alpha * (reward + self.gamma * max_future - old_val)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self.save()

    def save(self, filepath="rl_q_table.pkl"):
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            pass

    def load(self, filepath="rl_q_table.pkl"):
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    self.q_table = pickle.load(f)
            except Exception:
                pass

    def get_stats(self):
        return {
            "q_table_size": len(self.q_table),
            "epsilon": round(self.epsilon, 4)
        }
