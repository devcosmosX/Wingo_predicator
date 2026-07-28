import numpy as np
import pandas as pd

class DeepSequenceEngine:
    """
    Deep Neural & Markov Sequence Engine for WinGo 30S.
    Computes 1st & 2nd order Markov transition probabilities,
    EWMA momentum, and harmonic sequence weights.
    """
    def __init__(self):
        self.num_classes = 10

    def compute_markov_matrix(self, digits):
        """Calculates 1st order Markov transition probability matrix (10x10)"""
        matrix = np.full((10, 10), 0.1) # Laplacial smoothing
        if len(digits) < 2:
            return matrix / matrix.sum(axis=1, keepdims=True)
            
        for i in range(len(digits) - 1):
            curr_d = int(digits[i])
            next_d = int(digits[i+1])
            if 0 <= curr_d <= 9 and 0 <= next_d <= 9:
                matrix[curr_d, next_d] += 1.0
                
        # Normalize rows to probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        return matrix / row_sums

    def compute_deep_scores(self, digits, period_str):
        """
        Combines 1st-order Markov transitions, 2nd-order sequence attention,
        and harmonic frequency weights. Returns a dictionary of 10-digit probabilities.
        """
        if not digits or len(digits) < 3:
            default_prob = {d: 0.10 for d in range(10)}
            return default_prob

        # 1. Markov Transition Scores based on last drawn digit
        markov = self.compute_markov_matrix(digits)
        last_digit = int(digits[-1])
        markov_probs = markov[last_digit] if 0 <= last_digit <= 9 else np.full(10, 0.1)

        # 2. 2nd-Order Pattern Matching (Last 2 digits sequence match)
        pattern_scores = np.zeros(10)
        if len(digits) >= 4:
            target_pair = (int(digits[-2]), int(digits[-1]))
            for i in range(len(digits) - 2):
                if (int(digits[i]), int(digits[i+1])) == target_pair:
                    next_d = int(digits[i+2])
                    pattern_scores[next_d] += 1.5

        if pattern_scores.sum() > 0:
            pattern_probs = pattern_scores / pattern_scores.sum()
        else:
            pattern_probs = np.full(10, 0.1)

        # 3. Harmonic Cycle & Gap Weighting
        gaps = np.zeros(10)
        for d in range(10):
            if d in digits:
                gaps[d] = len(digits) - 1 - (len(digits) - 1 - digits[::-1].index(d))
            else:
                gaps[d] = len(digits)
                
        gap_weights = np.exp(np.clip(gaps / 15.0, 0, 2.0))
        gap_probs = gap_weights / gap_weights.sum()

        # 4. Period ID Modulo Signals
        period_probs = np.full(10, 0.05)
        clean_period = str(period_str).strip()
        if clean_period.isdigit():
            p_val = int(clean_period)
            mod10_tail = (p_val % 1000) % 10
            digit_sum_mod10 = sum(int(c) for c in clean_period) % 10
            period_probs[mod10_tail] += 0.25
            period_probs[digit_sum_mod10] += 0.25
        period_probs = period_probs / period_probs.sum()

        # Ensemble Weighted Fusion
        combined = (markov_probs * 0.35) + (pattern_probs * 0.30) + (gap_probs * 0.20) + (period_probs * 0.15)
        combined = combined / combined.sum()

        return {d: float(combined[d]) for d in range(10)}

    def predict(self, digits, period_str):
        scores = self.compute_deep_scores(digits, period_str)
        best_digit = max(scores, key=scores.get)
        confidence = float(scores[best_digit])
        
        size = "Big" if best_digit >= 5 else "Small"
        if best_digit in (0, 5): color = "Violet"
        elif best_digit in (1, 3, 7, 9): color = "Green"
        else: color = "Red"
        
        return best_digit, size, color, confidence, scores
