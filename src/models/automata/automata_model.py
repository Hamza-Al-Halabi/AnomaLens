import numpy as np
import json
from collections import defaultdict
try:
    from .sax import series_to_sax_patterns, build_sax_vocabulary
    from .levenshtein import resolve_pattern
except ImportError:
    from sax import series_to_sax_patterns, build_sax_vocabulary
    from levenshtein import resolve_pattern


class ProbabilisticAutomata:
    """
    Probabilistic Automata model for time series anomaly detection.

    Pipeline:
        PC1 series → SAX patterns (sliding window) → transition matrix
        → path probability → anomaly decision

    Training (fit):
        - Builds SAX vocabulary from training patterns
        - Counts state transitions
        - Computes transition probabilities with Laplace smoothing

    Inference (predict):
        - Maps unseen patterns via Levenshtein distance
        - Computes path probability for each window
        - Flags low-probability paths as anomalies
    """

    def __init__(self, config: dict):
        self.window_size   = config["automata"]["window_size"]
        self.alphabet_size = config["automata"]["alphabet_size"]
        self.smoothing     = config["automata"]["smoothing"]

        self.vocabulary        = set()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs  = {}   # {from_state: {to_state: prob}}
        self.threshold         = None # set during calibration on val set
        self._fitted           = False

    # ── Training ──────────────────────────────────────────────

    def fit(self, pc1_train: np.ndarray) -> "ProbabilisticAutomata":
        """
        Builds vocabulary and transition matrix from training PC1 series.
        Must be called on training data ONLY.
        """
        pc1_train = np.array(pc1_train).flatten()
        patterns  = series_to_sax_patterns(
            pc1_train, self.window_size, self.alphabet_size
        )

        self.vocabulary = build_sax_vocabulary(patterns)

        # Count transitions: pattern[i] → pattern[i+1]
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(patterns) - 1):
            src = patterns[i]
            dst = patterns[i + 1]
            self.transition_counts[src][dst] += 1

        # Compute probabilities with Laplace smoothing
        self._compute_transition_probs()
        self._fitted = True

        n_states     = len(self.vocabulary)
        n_transitions = sum(
            len(v) for v in self.transition_counts.values()
        )
        print(f"[Automata] Fitted: {n_states} states, "
              f"{n_transitions} unique transitions, "
              f"{len(patterns)} training patterns")

        return self

    def _compute_transition_probs(self):
        """
        P(Si → Sj) = (count(Si→Sj) + smoothing) /
                     (total_exits(Si) + smoothing * |vocab|)
        """
        self.transition_probs = {}
        vocab_size = len(self.vocabulary)

        for src, dst_counts in self.transition_counts.items():
            total = sum(dst_counts.values())
            self.transition_probs[src] = {}
            for dst, count in dst_counts.items():
                self.transition_probs[src][dst] = (
                    (count + self.smoothing) /
                    (total + self.smoothing * vocab_size)
                )

    def get_transition_prob(self, src: str, dst: str) -> float:
        """Returns transition probability P(src → dst)."""
        if src in self.transition_probs:
            return self.transition_probs[src].get(dst, self.smoothing)
        return self.smoothing

    # ── Calibration ───────────────────────────────────────────

    def calibrate_threshold(self,
                             pc1_val: np.ndarray,
                             y_val: np.ndarray,
                             percentile: float = 10.0):
        """
        Sets anomaly threshold using validation set path probabilities.
        Anomalies are in the bottom `percentile`% of path probabilities.
        """
        probs, _ = self._compute_path_probs(pc1_val)
        self.threshold = float(np.percentile(probs, percentile))
        print(f"[Automata] Threshold set to {self.threshold:.6f} "
              f"(p{percentile} of val path probs)")

    # ── Inference ─────────────────────────────────────────────

    def predict(self, pc1_test: np.ndarray) -> tuple:
        """
        Predicts anomaly labels for a test PC1 series.

        Returns:
            y_pred : np.ndarray of 0/1 labels (length = len(pc1_test))
            path_probs : np.ndarray of path probabilities per window
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        if self.threshold is None:
            raise RuntimeError("Call calibrate_threshold() before predict().")

        probs, _ = self._compute_path_probs(pc1_test)

        # Anomaly = below threshold
        window_labels = (probs < self.threshold).astype(int)

        # Map window-level labels back to original series length
        # Each window covers positions [i, i+window_size)
        # We use max-vote: a point is anomaly if any covering window is anomaly
        n = len(pc1_test)
        y_pred = np.zeros(n, dtype=int)
        for i, label in enumerate(window_labels):
            y_pred[i: i + self.window_size] = np.maximum(
                y_pred[i: i + self.window_size], label
            )

        return y_pred, probs

    def _compute_path_probs(self, pc1: np.ndarray) -> tuple:
        """
        Computes path probability for each consecutive pair of windows.
        Returns per-window probabilities and explanations.
        """
        pc1 = np.array(pc1).flatten()
        patterns = series_to_sax_patterns(
            pc1, self.window_size, self.alphabet_size
        )

        probs        = []
        explanations = []

        for i in range(len(patterns) - 1):
            raw_src = patterns[i]
            raw_dst = patterns[i + 1]

            src, src_status, src_dist = resolve_pattern(raw_src, self.vocabulary)
            dst, dst_status, dst_dist = resolve_pattern(raw_dst, self.vocabulary)

            prob = self.get_transition_prob(src, dst)
            probs.append(prob)

            explanations.append({
                "time_step"  : i,
                "state"      : src,
                "pattern"    : raw_src,
                "status"     : src_status,
                "mapped_to"  : src if src_status == "seen" else src,
                "next_state" : dst,
                "next_pattern": raw_dst,
                "next_status": dst_status,
                "probability": prob,
                "decision"   : "anomaly" if prob < (self.threshold or 0) else "normal"
            })

        # Pad last window (no transition available)
        if probs:
            probs.append(probs[-1])
            explanations.append(explanations[-1])

        return np.array(probs), explanations

    def explain(self, pc1_test: np.ndarray, time_step: int) -> dict:
        """
        Returns full explanation for a single time step.
        Format matches project spec Section X.F.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before explain().")

        pc1_test = np.array(pc1_test).flatten()
        patterns = series_to_sax_patterns(
            pc1_test, self.window_size, self.alphabet_size
        )

        if time_step >= len(patterns) - 1:
            raise ValueError(f"time_step {time_step} out of range.")

        raw_src = patterns[time_step]
        raw_dst = patterns[time_step + 1]

        src, src_status, src_dist = resolve_pattern(raw_src, self.vocabulary)
        dst, dst_status, dst_dist = resolve_pattern(raw_dst, self.vocabulary)

        prob     = self.get_transition_prob(src, dst)
        decision = "anomaly" if (self.threshold and prob < self.threshold) else "normal"

        explanation = {
            "time_step"   : time_step,
            "state"       : src,
            "pattern"     : raw_src,
            "status"      : src_status,
            "mapped_to"   : src,
            "levenshtein_distance": src_dist,
            "next_state"  : dst,
            "next_pattern": raw_dst,
            "next_status" : dst_status,
            "transitions" : {f"{src} -> {dst}": prob},
            "probability" : prob,
            "decision"    : decision,
            "confidence"  : prob,
            "threshold"   : self.threshold
        }

        return explanation

    def get_transition_matrix_df(self):
        """Returns transition probabilities as a pandas DataFrame (for heatmap)."""
        import pandas as pd
        states = sorted(self.vocabulary)
        matrix = pd.DataFrame(0.0, index=states, columns=states)
        for src, dst_probs in self.transition_probs.items():
            if src in states:
                for dst, prob in dst_probs.items():
                    if dst in states:
                        matrix.loc[src, dst] = prob
        return matrix

    def get_stats(self) -> dict:
        """Returns model statistics for reporting."""
        n_states = len(self.vocabulary)
        n_transitions = sum(len(v) for v in self.transition_probs.values())
        densities = [
            len(v) / max(n_states, 1)
            for v in self.transition_probs.values()
        ]
        return {
            "n_states"          : n_states,
            "n_transitions"     : n_transitions,
            "transition_density": float(np.mean(densities)) if densities else 0.0,
            "window_size"       : self.window_size,
            "alphabet_size"     : self.alphabet_size,
            "threshold"         : self.threshold
        }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))

    from src.data.loader import (load_config, load_batadal,
                                  get_batadal_features_target)
    from src.data.splitter import get_batadal_split
    from src.data.preprocessor import Preprocessor

    config = load_config()

    print("=" * 55)
    print("Automata model — BATADAL quick test")

    batadal_df = load_batadal(config)
    X_bat, y_bat = get_batadal_features_target(batadal_df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(
        X_bat, y_bat, config
    )

    # Preprocess — fit on train only
    prep = Preprocessor(config)
    _, X_train_pca = prep.fit_transform(X_train)
    _, X_val_pca   = prep.transform(X_val)
    _, X_test_pca  = prep.transform(X_test)

    pc1_train = X_train_pca[:, 0]
    pc1_val   = X_val_pca[:, 0]
    pc1_test  = X_test_pca[:, 0]

    # Build and run automata
    automata = ProbabilisticAutomata(config)
    automata.fit(pc1_train)
    automata.calibrate_threshold(pc1_val, y_val.values)

    y_pred, probs = automata.predict(pc1_test)

    from sklearn.metrics import classification_report
    # Align lengths
    min_len = min(len(y_test), len(y_pred))
    print("\n[Automata] Classification Report (BATADAL test set):")
    print(classification_report(y_test[:min_len], y_pred[:min_len],
                                 target_names=["normal", "anomaly"]))

    # Explain one step
    print("\n[Automata] Example explanation (time_step=5):")
    exp = automata.explain(pc1_test, time_step=5)
    print(json.dumps(exp, indent=2))

    # Stats
    print("\n[Automata] Model stats:")
    print(json.dumps(automata.get_stats(), indent=2))