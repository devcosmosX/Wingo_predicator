"""
Q-Learning Reinforcement Agent v2.0
=====================================
Fixes:
  - Q-table is NEVER mutated during predict() (read-only copy)
  - Experience replay buffer for stable convergence
  - Double Q-learning to reduce overestimation bias
  - Better state abstraction (quantized trend + gap features)
"""

import numpy as np
import os
import pickle
from collections import deque


class RLAgent:
    def __init__(self, alpha=0.15, gamma=0.85, epsilon=0.10, min_epsilon=0.02, decay=0.9995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        # Double Q-Learning: two Q-tables to reduce overestimation
        self.q_table_a = {}
        self.q_table_b = {}

        # Experience replay buffer (state, action, reward)
        self.replay_buffer = deque(maxlen=300)
        self.replay_batch_size = 16

        self.load()

    # ------------------------------------------------------------------
    # State hashing: quantized features for better generalization
    # ------------------------------------------------------------------
    def get_state_hash(self, last_digits):
        """
        State = (last_3_digits, trend_direction, mean_bucket)
        Using quantized features instead of raw values for better
        generalization with limited training data.
        """
        if len(last_digits) < 3:
            return ",".join(str(d) for d in last_digits)

        d1, d2, d3 = int(last_digits[-3]), int(last_digits[-2]), int(last_digits[-1])

        # Trend: +1 if rising, -1 if falling, 0 if flat
        trend = 0
        if d3 > d2:
            trend = 1
        elif d3 < d2:
            trend = -1

        # Mean bucket: low (0-3), mid (4-6), high (7-9)
        mean_val = (d1 + d2 + d3) / 3.0
        if mean_val < 3.5:
            bucket = "L"
        elif mean_val < 6.5:
            bucket = "M"
        else:
            bucket = "H"

        return f"{d1},{d2},{d3}|t{trend}|{bucket}"

    # ------------------------------------------------------------------
    # Predict: read-only — NEVER mutate Q-tables
    # ------------------------------------------------------------------
    def predict(self, last_digits, ml_pred=None, ml_conf=None):
        state = self.get_state_hash(last_digits)

        # Average both Q-tables for action selection (Double Q-Learning)
        qa = self.q_table_a.get(state, np.zeros(10))
        qb = self.q_table_b.get(state, np.zeros(10))
        q_avg = (qa + qb) / 2.0  # read-only average, no mutation

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(10))
            mode = "rl_explore"
        else:
            action = int(np.argmax(q_avg))
            mode = "rl_exploit"

        return action, mode, state

    # ------------------------------------------------------------------
    # Learn: update Q-tables with experience replay
    # ------------------------------------------------------------------
    def learn(self, state, action, actual_digit):
        # Calculate reward
        reward = self._compute_reward(action, actual_digit)

        # Store experience
        self.replay_buffer.append((state, int(action), reward))

        # Update Q-tables with current experience
        self._update_q(state, int(action), reward)

        # Experience replay: learn from random past experiences
        if len(self.replay_buffer) >= self.replay_batch_size:
            indices = np.random.choice(
                len(self.replay_buffer),
                size=min(self.replay_batch_size, len(self.replay_buffer)),
                replace=False,
            )
            for idx in indices:
                s, a, r = self.replay_buffer[idx]
                self._update_q(s, a, r)

        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self.save()

    def _compute_reward(self, action, actual_digit):
        """Asymmetric multi-target reward with partial credit."""
        action = int(action)
        actual_digit = int(actual_digit)

        if action == actual_digit:
            return 5.0  # exact digit match (rare, big reward)

        reward = 0.0

        # Size match partial credit
        pred_big = action >= 5
        actual_big = actual_digit >= 5
        if pred_big == actual_big:
            reward += 1.5
        else:
            reward -= 0.3

        # Color match partial credit
        color_map = {0: "V", 1: "G", 2: "R", 3: "G", 4: "R", 5: "V", 6: "R", 7: "G", 8: "R", 9: "G"}
        if color_map.get(action) == color_map.get(actual_digit):
            reward += 1.5
        else:
            reward -= 0.3

        # Proximity bonus (close digit = smaller penalty)
        dist = abs(action - actual_digit)
        if dist <= 1:
            reward += 0.5

        return reward

    def _update_q(self, state, action, reward):
        """Double Q-Learning update: randomly update Q_a or Q_b."""
        if state not in self.q_table_a:
            self.q_table_a[state] = np.zeros(10)
        if state not in self.q_table_b:
            self.q_table_b[state] = np.zeros(10)

        if np.random.rand() < 0.5:
            # Update Q_a using Q_b for next-state value
            best_a = int(np.argmax(self.q_table_a[state]))
            max_future = self.q_table_b[state][best_a]
            old_val = self.q_table_a[state][action]
            self.q_table_a[state][action] = old_val + self.alpha * (
                reward + self.gamma * max_future - old_val
            )
        else:
            # Update Q_b using Q_a for next-state value
            best_b = int(np.argmax(self.q_table_b[state]))
            max_future = self.q_table_a[state][best_b]
            old_val = self.q_table_b[state][action]
            self.q_table_b[state][action] = old_val + self.alpha * (
                reward + self.gamma * max_future - old_val
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath="rl_q_table.pkl"):
        try:
            data = {
                "q_table_a": self.q_table_a,
                "q_table_b": self.q_table_b,
                "epsilon": self.epsilon,
                "replay_buffer": list(self.replay_buffer),
            }
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def load(self, filepath="rl_q_table.pkl"):
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "q_table_a" in data:
                    self.q_table_a = data["q_table_a"]
                    self.q_table_b = data.get("q_table_b", {})
                    self.epsilon = data.get("epsilon", self.epsilon)
                    buf = data.get("replay_buffer", [])
                    self.replay_buffer = deque(buf, maxlen=300)
                elif isinstance(data, dict):
                    # Migrate from old single Q-table format
                    self.q_table_a = data
                    self.q_table_b = {}
            except Exception:
                pass

    def get_stats(self):
        total_states = len(set(list(self.q_table_a.keys()) + list(self.q_table_b.keys())))
        return {
            "q_table_size": total_states,
            "epsilon": round(self.epsilon, 4),
            "replay_buffer_size": len(self.replay_buffer),
        }
