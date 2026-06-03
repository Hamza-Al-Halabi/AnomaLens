"""
Statistical significance tests for model comparison.
- Wilcoxon signed-rank test: compares per-seed/per-fold F1 distributions
- McNemar test: compares paired predictions of two models
"""

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def wilcoxon_test(scores_a: list, scores_b: list, name_a="A", name_b="B") -> dict:
    """
    Wilcoxon signed-rank test on paired per-seed/per-fold F1 scores.
    Tests whether the median difference between two models is non-zero.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)

    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    # If all differences are zero, the test is undefined
    if np.all(a == b):
        return {
            "test"      : "wilcoxon",
            "comparison": f"{name_a} vs {name_b}",
            "statistic" : None,
            "p_value"   : 1.0,
            "significant": False,
            "note"      : "identical scores — no difference",
        }

    try:
        stat, p = wilcoxon(a, b)
    except ValueError as e:
        return {
            "test"      : "wilcoxon",
            "comparison": f"{name_a} vs {name_b}",
            "statistic" : None,
            "p_value"   : None,
            "significant": False,
            "note"      : str(e),
        }

    return {
        "test"       : "wilcoxon",
        "comparison" : f"{name_a} vs {name_b}",
        "mean_a"     : float(a.mean()),
        "mean_b"     : float(b.mean()),
        "statistic"  : float(stat),
        "p_value"    : float(p),
        "significant": bool(p < 0.05),
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a="A", name_b="B") -> dict:
    """
    McNemar test on paired predictions.
    Builds a 2x2 contingency table of where models agree/disagree
    relative to ground truth, then tests for significant difference.
    """
    y_true   = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    n = min(len(y_true), len(y_pred_a), len(y_pred_b))
    y_true, y_pred_a, y_pred_b = y_true[:n], y_pred_a[:n], y_pred_b[:n]

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    both_correct = np.sum(correct_a & correct_b)
    a_only       = np.sum(correct_a & ~correct_b)
    b_only       = np.sum(~correct_a & correct_b)
    both_wrong   = np.sum(~correct_a & ~correct_b)

    table = [[both_correct, a_only],
             [b_only,       both_wrong]]

    # exact=True if discordant pairs are small
    use_exact = (a_only + b_only) < 25
    result = mcnemar(table, exact=use_exact)

    return {
        "test"       : "mcnemar",
        "comparison" : f"{name_a} vs {name_b}",
        "contingency": {
            "both_correct": int(both_correct),
            f"{name_a}_only_correct": int(a_only),
            f"{name_b}_only_correct": int(b_only),
            "both_wrong"  : int(both_wrong),
        },
        "statistic"  : float(result.statistic),
        "p_value"    : float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.abspath("."))
    from src.data.loader import load_config

    config = load_config()

    # Load saved results
    log_dir = config["paths"]["logs"]
    with open(os.path.join(log_dir, "all_results.json")) as f:
        results = json.load(f)

    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS (Wilcoxon signed-rank)")
    print("=" * 60)

    output = {}
    for dataset in ["SKAB", "BATADAL"]:
        print(f"\n── {dataset} ──")
        output[dataset] = {}
        models = ["lstm", "gru", "cnn", "automata"]

        # Compare each DL model against automata
        for dl in ["lstm", "gru", "cnn"]:
            scores_dl   = results[dataset][dl]["per_seed_f1"]
            scores_auto = results[dataset]["automata"]["per_seed_f1"]
            test = wilcoxon_test(scores_dl, scores_auto, dl, "automata")
            output[dataset][f"{dl}_vs_automata"] = test
            sig = "✓ SIGNIFICANT" if test["significant"] else "✗ not significant"
            print(f"  {dl:6} vs automata: p={test['p_value']}  {sig}")

    # Save
    out_path = os.path.join(log_dir, "statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Logger] Saved → {out_path}")