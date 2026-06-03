"""
Probabilistic Explainability Module (Project Section X).

Produces deterministic, reproducible explanations for each decision of the
ProbabilisticAutomata model, including:
  - current state, observed pattern, seen/unseen status
  - Levenshtein mapping for unseen patterns
  - state transitions and their probabilities
  - path probability (product of consecutive transition probabilities)
  - final decision and confidence score
"""

import json
import numpy as np
try:
    from src.models.automata.sax import series_to_sax_patterns
    from src.models.automata.levenshtein import resolve_pattern
except ModuleNotFoundError:
    # When running this file directly as a script, the package root may
    # not be on sys.path. Add the project src root (two levels up) and
    # retry imports so the module can be executed standalone.
    import os, sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models.automata.sax import series_to_sax_patterns
    from src.models.automata.levenshtein import resolve_pattern


class AutomataExplainer:
    """Wraps a fitted ProbabilisticAutomata to generate explanations."""

    def __init__(self, automata):
        if not automata._fitted:
            raise RuntimeError("Automata must be fitted before explaining.")
        self.automata = automata
        self.ws = automata.window_size
        self.al = automata.alphabet_size

    def explain_step(self, pc1: np.ndarray, t: int, path_len: int = 2) -> dict:
        """
        Explains the decision at time step t.
        Path probability = product of `path_len` consecutive transition probs
        starting from t (matches spec example: 0.72 * 0.15 = 0.108).
        """
        pc1      = np.array(pc1).flatten()
        patterns = series_to_sax_patterns(pc1, self.ws, self.al)

        if t + path_len >= len(patterns):
            path_len = max(1, len(patterns) - t - 1)

        transitions   = {}
        path_prob     = 1.0
        first_status  = None
        first_mapped  = None
        first_dist    = None
        first_pattern = patterns[t] if t < len(patterns) else None

        for i in range(t, t + path_len):
            raw_src = patterns[i]
            raw_dst = patterns[i + 1]

            src, src_status, src_dist = resolve_pattern(raw_src, self.automata.vocabulary)
            dst, _, _                 = resolve_pattern(raw_dst, self.automata.vocabulary)

            prob = self.automata.get_transition_prob(src, dst)
            transitions[f"{src} -> {dst}"] = round(prob, 4)
            path_prob *= prob

            if i == t:
                first_status = src_status
                first_mapped = src
                first_dist   = src_dist

        threshold = self.automata.threshold or 0.0
        decision  = "anomaly" if path_prob < threshold else "normal"
        conf_label = "Low" if decision == "anomaly" else "High"

        explanation = {
            "time_step"           : t,
            "state"               : first_mapped,
            "pattern"             : first_pattern,
            "status"              : first_status,
            "mapped_to"           : first_mapped if first_status == "unseen" else first_pattern,
            "levenshtein_distance": first_dist,
            "transitions"         : transitions,
            "path_probability"    : round(path_prob, 6),
            "threshold"           : round(threshold, 6),
            "decision"            : decision,
            "confidence_score"    : round(path_prob, 6),
            "confidence_label"    : conf_label,
        }
        return explanation

    def explain_spec_format(self, pc1: np.ndarray, t: int) -> str:
        """Returns the human-readable [SYSTEM DECISION] format from spec X.E."""
        exp = self.explain_step(pc1, t)
        lines = [
            "[SYSTEM DECISION]",
            f"Time Step: t = {exp['time_step']}",
            f"Previous State: \"{exp['state']}\"",
            f"Incoming Pattern: \"{exp['pattern']}\"",
            f"Status: {exp['status'].capitalize()}",
        ]
        if exp["status"] == "unseen":
            lines.append(f"Nearest Pattern: \"{exp['mapped_to']}\" "
                         f"(distance = {exp['levenshtein_distance']})")
        lines.append("Transitions:")
        for trans, prob in exp["transitions"].items():
            lines.append(f"  {trans} : {prob}")
        prob_str = " * ".join(str(p) for p in exp["transitions"].values())
        lines.append("Path Probability:")
        lines.append(f"  {prob_str} = {exp['path_probability']}")
        lines.append("Decision:")
        lines.append(f"  {'Low' if exp['decision']=='anomaly' else 'High'} "
                     f"probability path detected")
        lines.append(f"Result: {exp['decision'].upper()}")
        lines.append(f"Confidence Score: {exp['confidence_score']} ({exp['confidence_label']})")
        return "\n".join(lines)

    def explain_json(self, pc1: np.ndarray, t: int) -> str:
        """Returns the mandatory JSON format from spec X.F."""
        exp = self.explain_step(pc1, t)
        compact = {
            "time_step" : exp["time_step"],
            "state"     : exp["state"],
            "pattern"   : exp["pattern"],
            "status"    : exp["status"],
            "mapped_to" : exp["mapped_to"],
            "probability": exp["path_probability"],
            "decision"  : exp["decision"],
        }
        return json.dumps(compact, indent=2)

    def explain_all(self, pc1: np.ndarray) -> list:
        """Returns explanations for all time steps (for table export)."""
        pc1      = np.array(pc1).flatten()
        patterns = series_to_sax_patterns(pc1, self.ws, self.al)
        return [self.explain_step(pc1, t) for t in range(len(patterns) - 2)]


if __name__ == "__main__":
    # Ensure project root is on sys.path when running as a script
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from src.data.loader       import load_config, load_batadal, get_batadal_features_target
    from src.data.splitter     import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.automata.automata_model import ProbabilisticAutomata

    config = load_config()

    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    _, X_tr_pca = prep.fit_transform(X_train)
    _, X_va_pca = prep.transform(X_val)
    _, X_te_pca = prep.transform(X_test)

    automata = ProbabilisticAutomata(config)
    automata.fit(X_tr_pca[:, 0])
    automata.calibrate_threshold(X_va_pca[:, 0], y_val.values)

    explainer = AutomataExplainer(automata)

    print("=" * 55)
    print("SPEC FORMAT (Section X.E):")
    print("=" * 55)
    print(explainer.explain_spec_format(X_te_pca[:, 0], t=5))

    print("\n" + "=" * 55)
    print("JSON FORMAT (Section X.F):")
    print("=" * 55)
    print(explainer.explain_json(X_te_pca[:, 0], t=5))

    # Find an anomaly example
    print("\n" + "=" * 55)
    print("Searching for an anomaly decision...")
    all_exp = explainer.explain_all(X_te_pca[:, 0])
    anomalies = [e for e in all_exp if e["decision"] == "anomaly"]
    print(f"Found {len(anomalies)} anomaly decisions out of {len(all_exp)} steps")
    if anomalies:
        print("\nExample anomaly explanation:")
        print(json.dumps(anomalies[0], indent=2))