"""
Deep Sequence & Markov Attention Engine v2.0
=============================================
Accepts FULL historical digit sequence (1000+ draws) and computes:
  1. 1st-order Markov transition matrix (last digit -> next)
  2. 2nd-order Markov (last 2 digits -> next)
  3. 3rd-order Markov (last 3 digits -> next)
  4. Exponentially-weighted Markov (recent transitions count more)
  5. Autocorrelation analysis at lags 1-30
  6. Frequency momentum (EWMA of digit occurrence rates)
  7. Gap-based overdue weighting

All outputs are calibrated probability distributions summing to 1.0.
"""

import numpy as np
from collections import defaultdict


class DeepSequenceEngine:
    def __init__(self):
        self.num_classes = 10

    # ------------------------------------------------------------------
    # 1st-order Markov: P(next | last_digit)
    # ------------------------------------------------------------------
    def _markov_order1(self, digits, decay=None):
        matrix = np.ones((10, 10)) * 0.01  # mild Laplace smoothing
        n = len(digits)
        if n < 2:
            return np.full(10, 0.1)

        for i in range(n - 1):
            c, nx = int(digits[i]), int(digits[i + 1])
            if 0 <= c <= 9 and 0 <= nx <= 9:
                w = 1.0
                if decay is not None:
                    w = decay ** (n - 2 - i)  # recent transitions weighted higher
                matrix[c, nx] += w

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

        last_d = int(digits[-1])
        if 0 <= last_d <= 9:
            return matrix[last_d]
        return np.full(10, 0.1)

    # ------------------------------------------------------------------
    # 2nd-order Markov: P(next | last_2_digits)
    # ------------------------------------------------------------------
    def _markov_order2(self, digits):
        trans = defaultdict(lambda: np.ones(10) * 0.01)
        n = len(digits)
        if n < 3:
            return np.full(10, 0.1)

        for i in range(n - 2):
            key = (int(digits[i]), int(digits[i + 1]))
            nx = int(digits[i + 2])
            if 0 <= nx <= 9:
                trans[key][nx] += 1.0

        query = (int(digits[-2]), int(digits[-1]))
        probs = trans[query].copy()
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.full(10, 0.1)
        return probs

    # ------------------------------------------------------------------
    # 3rd-order Markov: P(next | last_3_digits)
    # ------------------------------------------------------------------
    def _markov_order3(self, digits):
        trans = defaultdict(lambda: np.ones(10) * 0.01)
        n = len(digits)
        if n < 4:
            return np.full(10, 0.1)

        for i in range(n - 3):
            key = (int(digits[i]), int(digits[i + 1]), int(digits[i + 2]))
            nx = int(digits[i + 3])
            if 0 <= nx <= 9:
                trans[key][nx] += 1.0

        query = (int(digits[-3]), int(digits[-2]), int(digits[-1]))
        probs = trans[query].copy()
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.full(10, 0.1)
        return probs

    # ------------------------------------------------------------------
    # Autocorrelation: detect periodic patterns at lags 1..max_lag
    # ------------------------------------------------------------------
    def _autocorrelation_scores(self, digits, max_lag=30):
        n = len(digits)
        if n < max_lag + 10:
            return np.full(10, 0.1)

        arr = np.array(digits, dtype=float)
        mean = arr.mean()
        var = np.var(arr)
        if var < 1e-9:
            return np.full(10, 0.1)

        # Find lags with strongest autocorrelation
        best_lag = 1
        best_corr = 0.0
        for lag in range(1, min(max_lag + 1, n)):
            c = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean)) / var
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag

        # If we found a meaningful lag, use it for prediction
        scores = np.ones(10) * 0.05
        if abs(best_corr) > 0.05 and best_lag <= n:
            # The digit that appeared `best_lag` draws ago may repeat
            lagged_digit = int(digits[-best_lag])
            if 0 <= lagged_digit <= 9:
                boost = min(abs(best_corr) * 3.0, 1.5)
                scores[lagged_digit] += boost

        total = scores.sum()
        return scores / total if total > 0 else np.full(10, 0.1)

    # ------------------------------------------------------------------
    # Gap-based overdue weighting: digits not seen for a while get a boost
    # ------------------------------------------------------------------
    def _gap_scores(self, digits):
        n = len(digits)
        if n == 0:
            return np.full(10, 0.1)

        last_seen = {}
        for i, d in enumerate(digits):
            last_seen[int(d)] = i

        gaps = np.zeros(10)
        for d in range(10):
            if d in last_seen:
                gaps[d] = n - 1 - last_seen[d]
            else:
                gaps[d] = n

        # Soft overdue weighting (not exponential — that overweights outliers)
        weights = 1.0 + np.log1p(gaps / 10.0)
        total = weights.sum()
        return weights / total if total > 0 else np.full(10, 0.1)

    # ------------------------------------------------------------------
    # EWMA frequency momentum
    # ------------------------------------------------------------------
    def _ewma_frequency(self, digits, span=50):
        n = len(digits)
        if n == 0:
            return np.full(10, 0.1)

        alpha = 2.0 / (span + 1)
        freqs = np.zeros(10)
        for d in digits:
            freqs *= (1.0 - alpha)
            freqs[int(d)] += alpha

        total = freqs.sum()
        return freqs / total if total > 0 else np.full(10, 0.1)

    # ------------------------------------------------------------------
    # Main prediction: fuse all signals
    # ------------------------------------------------------------------
    def predict(self, digits, period_str=""):
        """
        Args:
            digits: list of ints — the FULL historical digit sequence (oldest first)
            period_str: the period ID string (used minimally — see note)

        Returns:
            (best_digit, size, color, confidence, score_dict)
        """
        if not digits or len(digits) < 3:
            uniform = {d: 0.10 for d in range(10)}
            return 5, "Big", "Violet", 0.10, uniform

        n = len(digits)

        # Compute all sub-model probability distributions
        p_m1 = self._markov_order1(digits)
        p_m1_decay = self._markov_order1(digits, decay=0.995)
        p_m2 = self._markov_order2(digits)
        p_m3 = self._markov_order3(digits) if n >= 4 else np.full(10, 0.1)
        p_auto = self._autocorrelation_scores(digits)
        p_gap = self._gap_scores(digits)
        p_ewma = self._ewma_frequency(digits)

        # Adaptive weighting: higher-order models get more weight when they
        # have enough data, less when sparse
        w_m1 = 0.20
        w_m1d = 0.15
        w_m2 = 0.20 if n >= 100 else 0.10
        w_m3 = 0.15 if n >= 300 else 0.05
        w_auto = 0.10
        w_gap = 0.10
        w_ewma = 0.10

        # Normalize weights to sum to 1
        total_w = w_m1 + w_m1d + w_m2 + w_m3 + w_auto + w_gap + w_ewma
        combined = (
            p_m1 * (w_m1 / total_w)
            + p_m1_decay * (w_m1d / total_w)
            + p_m2 * (w_m2 / total_w)
            + p_m3 * (w_m3 / total_w)
            + p_auto * (w_auto / total_w)
            + p_gap * (w_gap / total_w)
            + p_ewma * (w_ewma / total_w)
        )

        # Ensure valid probability distribution
        combined = np.clip(combined, 0, None)
        total = combined.sum()
        if total > 0:
            combined /= total
        else:
            combined = np.full(10, 0.1)

        best_digit = int(np.argmax(combined))
        confidence = float(combined[best_digit])

        size = "Big" if best_digit >= 5 else "Small"
        if best_digit in (0, 5):
            color = "Violet"
        elif best_digit in (1, 3, 7, 9):
            color = "Green"
        else:
            color = "Red"

        score_dict = {d: float(combined[d]) for d in range(10)}
        return best_digit, size, color, confidence, score_dict
