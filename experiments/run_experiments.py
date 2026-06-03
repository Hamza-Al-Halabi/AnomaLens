"""
AnomaLens — Additional Experiments
Runs noise robustness, unseen-pattern, cross-dataset, and parameter-sweep
experiments on top of the main pipeline results.

Usage:
    python experiments/run_experiments.py --exp all
    python experiments/run_experiments.py --exp noise
    python experiments/run_experiments.py --exp unseen
    python experiments/run_experiments.py --exp params
    python experiments/run_experiments.py --exp cross
    python experiments/run_experiments.py --exp stats
"""

import os
import sys
import json
import argparse
import numpy as np
import copy

sys.path.insert(0, os.path.abspath("."))

from src.data.loader       import (load_config, load_skab, load_batadal,
                                    get_skab_features_target,
                                    get_batadal_features_target)
from src.data.splitter     import get_skab_folds, get_batadal_split
from src.data.preprocessor import Preprocessor
from src.models.deep_learning.base  import train_model, predict_model, set_seed
from src.models.deep_learning.lstm  import LSTMModel
from src.models.deep_learning.gru   import GRUModel
from src.models.deep_learning.cnn1d import CNN1DModel
from src.models.automata.automata_model import ProbabilisticAutomata
from sklearn.metrics import f1_score, precision_score, recall_score


# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def save_json(data, config, filename):
    path = os.path.join(config["paths"]["logs"], filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Logger] Saved -> {path}")


def compute_metrics(y_true, y_pred):
    return {
        "f1"       : float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
    }


def make_dl_model(name, config, n_features):
    name = name.lower()
    if name == "lstm":   return LSTMModel(config, n_features)
    if name == "gru":    return GRUModel(config, n_features)
    if name == "cnn":    return CNN1DModel(config, n_features)
    raise ValueError(f"Unknown model: {name}")


# ── 1. Noise robustness ───────────────────────────────────────────────────────

def run_noise_experiment(config):
    """Adds Gaussian noise to test features and measures F1 degradation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Noise Robustness")
    print("=" * 60)

    noise_std = config["noise"]["gaussian_std"]
    seed      = config["model"]["seeds"][0]
    results   = {}

    # BATADAL
    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    X_tr_sc, X_tr_pca = prep.fit_transform(X_train)
    X_va_sc, X_va_pca = prep.transform(X_val)
    X_te_sc, X_te_pca = prep.transform(X_test)

    # Noisy test set
    rng = np.random.RandomState(seed)
    X_te_noisy   = X_te_sc  + rng.normal(0, noise_std, X_te_sc.shape)
    X_te_pca_noisy = X_te_pca + rng.normal(0, noise_std, X_te_pca.shape)

    results["BATADAL"] = {}

    for name in ["lstm", "gru", "cnn"]:
        model = make_dl_model(name, config, X_tr_sc.shape[1])
        _, trained = train_model(model, config, X_tr_sc, y_train.values,
                                  X_va_sc, y_val.values, seed=seed, model_name=name)
        clean = predict_model(trained, config, X_te_sc, y_test.values)
        noisy = predict_model(trained, config, X_te_noisy, y_test.values)
        results["BATADAL"][name] = {
            "clean_f1": clean["f1"],
            "noisy_f1": noisy["f1"],
            "f1_drop" : round(clean["f1"] - noisy["f1"], 4),
        }
        print(f"  {name.upper()} BATADAL: clean={clean['f1']:.4f}  noisy={noisy['f1']:.4f}  drop={results['BATADAL'][name]['f1_drop']:.4f}")

    automata = ProbabilisticAutomata(config)
    automata.fit(X_tr_pca[:, 0])
    automata.calibrate_threshold(X_va_pca[:, 0], y_val.values)
    yp_clean, _ = automata.predict(X_te_pca[:, 0])
    yp_noisy, _ = automata.predict(X_te_pca_noisy[:, 0])
    ml_c = min(len(y_test), len(yp_clean))
    ml_n = min(len(y_test), len(yp_noisy))
    f1_clean = float(f1_score(y_test.values[:ml_c], yp_clean[:ml_c], zero_division=0))
    f1_noisy = float(f1_score(y_test.values[:ml_n], yp_noisy[:ml_n], zero_division=0))
    results["BATADAL"]["automata"] = {
        "clean_f1": f1_clean,
        "noisy_f1": f1_noisy,
        "f1_drop" : round(f1_clean - f1_noisy, 4),
    }
    print(f"  AUTOMATA BATADAL: clean={f1_clean:.4f}  noisy={f1_noisy:.4f}  drop={results['BATADAL']['automata']['f1_drop']:.4f}")

    # SKAB (first fold only for speed)
    df_s = load_skab(config)
    X_s, y_s, groups_s = get_skab_features_target(df_s, config)
    folds = get_skab_folds(X_s, y_s, groups_s, config)
    tr_idx, te_idx = folds[0]

    X_tr_raw = X_s.iloc[tr_idx]
    X_te_raw = X_s.iloc[te_idx]
    y_tr     = y_s.iloc[tr_idx]
    y_te     = y_s.iloc[te_idx]
    n_val    = int(len(X_tr_raw) * 0.2)
    X_va_raw = X_tr_raw.iloc[-n_val:]
    X_tr_raw = X_tr_raw.iloc[:-n_val]
    y_va     = y_tr.iloc[-n_val:]
    y_tr     = y_tr.iloc[:-n_val]

    prep2 = Preprocessor(config)
    X_tr_sc2, X_tr_pca2 = prep2.fit_transform(X_tr_raw)
    X_va_sc2, X_va_pca2 = prep2.transform(X_va_raw)
    X_te_sc2, X_te_pca2 = prep2.transform(X_te_raw)

    rng2 = np.random.RandomState(seed)
    X_te_noisy2     = X_te_sc2  + rng2.normal(0, noise_std, X_te_sc2.shape)
    X_te_pca_noisy2 = X_te_pca2 + rng2.normal(0, noise_std, X_te_pca2.shape)

    results["SKAB"] = {}
    for name in ["lstm", "gru", "cnn"]:
        model = make_dl_model(name, config, X_tr_sc2.shape[1])
        _, trained = train_model(model, config, X_tr_sc2, y_tr.values,
                                  X_va_sc2, y_va.values, seed=seed, model_name=name)
        clean = predict_model(trained, config, X_te_sc2, y_te.values)
        noisy = predict_model(trained, config, X_te_noisy2, y_te.values)
        results["SKAB"][name] = {
            "clean_f1": clean["f1"],
            "noisy_f1": noisy["f1"],
            "f1_drop" : round(clean["f1"] - noisy["f1"], 4),
        }
        print(f"  {name.upper()} SKAB: clean={clean['f1']:.4f}  noisy={noisy['f1']:.4f}  drop={results['SKAB'][name]['f1_drop']:.4f}")

    automata2 = ProbabilisticAutomata(config)
    automata2.fit(X_tr_pca2[:, 0])
    automata2.calibrate_threshold(X_va_pca2[:, 0], y_va.values)
    yp_clean2, _ = automata2.predict(X_te_pca2[:, 0])
    yp_noisy2, _ = automata2.predict(X_te_pca_noisy2[:, 0])
    ml_c2 = min(len(y_te), len(yp_clean2))
    ml_n2 = min(len(y_te), len(yp_noisy2))
    f1_clean2 = float(f1_score(y_te.values[:ml_c2], yp_clean2[:ml_c2], zero_division=0))
    f1_noisy2 = float(f1_score(y_te.values[:ml_n2], yp_noisy2[:ml_n2], zero_division=0))
    results["SKAB"]["automata"] = {
        "clean_f1": f1_clean2,
        "noisy_f1": f1_noisy2,
        "f1_drop" : round(f1_clean2 - f1_noisy2, 4),
    }
    print(f"  AUTOMATA SKAB: clean={f1_clean2:.4f}  noisy={f1_noisy2:.4f}  drop={results['SKAB']['automata']['f1_drop']:.4f}")

    save_json(results, config, "experiments_noise.json")
    return results


# ── 2. Unseen patterns ────────────────────────────────────────────────────────

def run_unseen_experiment(config):
    """
    Measures how many test SAX patterns are 'unseen' (out-of-vocabulary)
    and the F1 impact of having high vs low unseen rates.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Unseen Pattern Analysis")
    print("=" * 60)

    from src.models.automata.sax import series_to_sax_patterns, build_sax_vocabulary
    from src.models.automata.levenshtein import find_nearest_pattern

    seed = config["model"]["seeds"][0]
    results = {}

    for dataset_name, loader_fn, splitter_fn in [
        ("BATADAL",
         lambda: (load_batadal(config), get_batadal_features_target, None),
         "batadal"),
        ("SKAB",
         lambda: (load_skab(config), get_skab_features_target, get_skab_folds),
         "skab"),
    ]:
        print(f"\n  -- {dataset_name} --")
        if dataset_name == "BATADAL":
            df = load_batadal(config)
            X, y = get_batadal_features_target(df, config)
            X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)
        else:
            df = load_skab(config)
            X, y, groups = get_skab_features_target(df, config)
            folds = get_skab_folds(X, y, groups, config)
            tr_idx, te_idx = folds[0]
            X_train = X.iloc[tr_idx]; y_train = y.iloc[tr_idx]
            X_test  = X.iloc[te_idx]; y_test  = y.iloc[te_idx]
            n_val   = int(len(X_train) * 0.2)
            X_val   = X_train.iloc[-n_val:]; y_val = y_train.iloc[-n_val:]
            X_train = X_train.iloc[:-n_val]; y_train = y_train.iloc[:-n_val]

        np.random.seed(seed)
        prep = Preprocessor(config)
        _, X_tr_pca = prep.fit_transform(X_train)
        _, X_va_pca = prep.transform(X_val)
        _, X_te_pca = prep.transform(X_test)

        automata = ProbabilisticAutomata(config)
        automata.fit(X_tr_pca[:, 0])
        automata.calibrate_threshold(X_va_pca[:, 0], y_val.values)
        y_pred, _ = automata.predict(X_te_pca[:, 0])

        # Compute unseen rate
        ws = config["automata"]["window_size"]
        al = config["automata"]["alphabet_size"]
        test_patterns  = series_to_sax_patterns(X_te_pca[:, 0], ws, al)
        train_vocab    = automata.vocabulary
        n_unseen       = sum(1 for p in test_patterns if p not in train_vocab)
        unseen_rate    = n_unseen / max(len(test_patterns), 1)

        # Average Levenshtein distance for unseen patterns
        lev_dists = []
        for p in test_patterns:
            if p not in train_vocab:
                _, dist = find_nearest_pattern(p, train_vocab)
                lev_dists.append(dist)
        avg_lev = float(np.mean(lev_dists)) if lev_dists else 0.0

        ml    = min(len(y_test), len(y_pred))
        f1    = float(f1_score(y_test.values[:ml], y_pred[:ml], zero_division=0))

        results[dataset_name] = {
            "vocab_size"    : len(train_vocab),
            "test_patterns" : len(test_patterns),
            "unseen_count"  : int(n_unseen),
            "unseen_rate"   : round(unseen_rate, 4),
            "avg_lev_dist"  : round(avg_lev, 4),
            "f1"            : round(f1, 4),
        }
        print(f"  Vocab size: {len(train_vocab)} | Unseen rate: {unseen_rate:.2%} | "
              f"Avg Lev dist: {avg_lev:.2f} | F1: {f1:.4f}")

    save_json(results, config, "experiments_unseen.json")
    return results


# ── 3. Parameter sensitivity ──────────────────────────────────────────────────

def run_params_experiment(config):
    """Parameter sweep: window_size and alphabet_size on BATADAL."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Automata Parameter Sensitivity (BATADAL)")
    print("=" * 60)

    seed   = config["model"]["seeds"][0]
    w_vals = config["automata"]["param_sweep"]["window_sizes"]
    a_vals = config["automata"]["param_sweep"]["alphabet_sizes"]

    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    np.random.seed(seed)
    prep = Preprocessor(config)
    _, X_tr_pca = prep.fit_transform(X_train)
    _, X_va_pca = prep.transform(X_val)
    _, X_te_pca = prep.transform(X_test)

    results = {"window_size": {}, "alphabet_size": {}}

    # Sweep window_size (keep alphabet=3)
    print("\n  Window size sweep (alphabet=3):")
    for ws in w_vals:
        cfg_sweep = copy.deepcopy(config)
        cfg_sweep["automata"]["window_size"]   = ws
        cfg_sweep["automata"]["alphabet_size"] = 3
        auto = ProbabilisticAutomata(cfg_sweep)
        auto.fit(X_tr_pca[:, 0])
        auto.calibrate_threshold(X_va_pca[:, 0], y_val.values)
        y_pred, _ = auto.predict(X_te_pca[:, 0])
        ml = min(len(y_test), len(y_pred))
        f1 = float(f1_score(y_test.values[:ml], y_pred[:ml], zero_division=0))
        stats = auto.get_stats()
        results["window_size"][str(ws)] = {
            "f1"      : round(f1, 4),
            "n_states": stats["n_states"],
            "n_transitions": stats["n_transitions"],
        }
        print(f"    window={ws}: F1={f1:.4f}  states={stats['n_states']}")

    # Sweep alphabet_size (keep window=4)
    print("\n  Alphabet size sweep (window=4):")
    for al in a_vals:
        cfg_sweep = copy.deepcopy(config)
        cfg_sweep["automata"]["window_size"]   = 4
        cfg_sweep["automata"]["alphabet_size"] = al
        auto = ProbabilisticAutomata(cfg_sweep)
        auto.fit(X_tr_pca[:, 0])
        auto.calibrate_threshold(X_va_pca[:, 0], y_val.values)
        y_pred, _ = auto.predict(X_te_pca[:, 0])
        ml = min(len(y_test), len(y_pred))
        f1 = float(f1_score(y_test.values[:ml], y_pred[:ml], zero_division=0))
        stats = auto.get_stats()
        results["alphabet_size"][str(al)] = {
            "f1"      : round(f1, 4),
            "n_states": stats["n_states"],
            "n_transitions": stats["n_transitions"],
        }
        print(f"    alphabet={al}: F1={f1:.4f}  states={stats['n_states']}")

    save_json({"params": results}, config, "experiments_params.json")
    return results


# ── 4. Cross-dataset ──────────────────────────────────────────────────────────

def run_cross_dataset_experiment(config):
    """
    Train automata on SKAB (first fold), test on BATADAL,
    and vice versa — measures cross-domain generalisation.
    Uses PCA-compressed single feature since both datasets differ in dimensionality.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Cross-Dataset Generalisation (Automata)")
    print("=" * 60)

    seed = config["model"]["seeds"][0]
    np.random.seed(seed)

    # Load both datasets
    df_bat = load_batadal(config)
    X_bat, y_bat = get_batadal_features_target(df_bat, config)
    X_tr_b, X_va_b, X_te_b, y_tr_b, y_va_b, y_te_b = get_batadal_split(X_bat, y_bat, config)

    df_skab = load_skab(config)
    X_sk, y_sk, groups_sk = get_skab_features_target(df_skab, config)
    folds = get_skab_folds(X_sk, y_sk, groups_sk, config)
    tr_idx, te_idx = folds[0]
    X_tr_s = X_sk.iloc[tr_idx]; y_tr_s = y_sk.iloc[tr_idx]
    X_te_s = X_sk.iloc[te_idx]; y_te_s = y_sk.iloc[te_idx]
    n_val  = int(len(X_tr_s) * 0.2)
    X_va_s = X_tr_s.iloc[-n_val:]; y_va_s = y_tr_s.iloc[-n_val:]
    X_tr_s = X_tr_s.iloc[:-n_val]; y_tr_s = y_tr_s.iloc[:-n_val]

    results = {}

    # Train on BATADAL, test on SKAB
    prep_bat = Preprocessor(config)
    _, pc_tr_b  = prep_bat.fit_transform(X_tr_b)
    _, pc_te_b  = prep_bat.transform(X_te_b)
    _, pc_va_b  = prep_bat.transform(X_va_b)

    # Need to re-compress SKAB through BATADAL's PCA (same 1D projection)
    # Both use PCA(1), so we fit on BATADAL train and apply to SKAB
    prep_cross_bs = Preprocessor(config)
    prep_cross_bs.fit_transform(X_tr_b)  # fit on BATADAL
    try:
        _, pc_skab_via_bat = prep_cross_bs.transform(X_te_s)
        cross_possible_bs = True
    except Exception:
        cross_possible_bs = False

    if cross_possible_bs:
        auto_b = ProbabilisticAutomata(config)
        auto_b.fit(pc_tr_b[:, 0])
        auto_b.calibrate_threshold(pc_va_b[:, 0], y_va_b.values)
        y_pred_cross, _ = auto_b.predict(pc_skab_via_bat[:, 0])
        ml = min(len(y_te_s), len(y_pred_cross))
        f1_cross = float(f1_score(y_te_s.values[:ml], y_pred_cross[:ml], zero_division=0))
        print(f"  Train=BATADAL -> Test=SKAB: F1={f1_cross:.4f}")
        results["batadal_to_skab"] = {"f1": round(f1_cross, 4)}
    else:
        results["batadal_to_skab"] = {"f1": None, "note": "feature_dim_mismatch"}

    # Train on SKAB, test on BATADAL
    prep_skab = Preprocessor(config)
    _, pc_tr_s  = prep_skab.fit_transform(X_tr_s)
    _, pc_va_s  = prep_skab.transform(X_va_s)

    prep_cross_sb = Preprocessor(config)
    prep_cross_sb.fit_transform(X_tr_s)  # fit on SKAB
    try:
        _, pc_bat_via_skab = prep_cross_sb.transform(X_te_b)
        cross_possible_sb = True
    except Exception:
        cross_possible_sb = False

    if cross_possible_sb:
        auto_s = ProbabilisticAutomata(config)
        auto_s.fit(pc_tr_s[:, 0])
        auto_s.calibrate_threshold(pc_va_s[:, 0], y_va_s.values)
        y_pred_cross2, _ = auto_s.predict(pc_bat_via_skab[:, 0])
        ml2 = min(len(y_te_b), len(y_pred_cross2))
        f1_cross2 = float(f1_score(y_te_b.values[:ml2], y_pred_cross2[:ml2], zero_division=0))
        print(f"  Train=SKAB -> Test=BATADAL: F1={f1_cross2:.4f}")
        results["skab_to_batadal"] = {"f1": round(f1_cross2, 4)}
    else:
        results["skab_to_batadal"] = {"f1": None, "note": "feature_dim_mismatch"}

    # In-domain baselines for context
    auto_bat_id = ProbabilisticAutomata(config)
    auto_bat_id.fit(pc_tr_b[:, 0])
    auto_bat_id.calibrate_threshold(pc_va_b[:, 0], y_va_b.values)
    yp, _ = auto_bat_id.predict(pc_te_b[:, 0])
    ml3 = min(len(y_te_b), len(yp))
    f1_id_bat = float(f1_score(y_te_b.values[:ml3], yp[:ml3], zero_division=0))

    prep_skab2 = Preprocessor(config)
    _, pc_tr_s2 = prep_skab2.fit_transform(X_tr_s)
    _, pc_va_s2 = prep_skab2.transform(X_va_s)
    _, pc_te_s2 = prep_skab2.transform(X_te_s)
    auto_sk_id = ProbabilisticAutomata(config)
    auto_sk_id.fit(pc_tr_s2[:, 0])
    auto_sk_id.calibrate_threshold(pc_va_s2[:, 0], y_va_s.values)
    yp2, _ = auto_sk_id.predict(pc_te_s2[:, 0])
    ml4 = min(len(y_te_s), len(yp2))
    f1_id_sk = float(f1_score(y_te_s.values[:ml4], yp2[:ml4], zero_division=0))

    results["in_domain_batadal"] = {"f1": round(f1_id_bat, 4)}
    results["in_domain_skab"]    = {"f1": round(f1_id_sk,  4)}
    print(f"  In-domain BATADAL: F1={f1_id_bat:.4f}")
    print(f"  In-domain SKAB:    F1={f1_id_sk:.4f}")

    save_json(results, config, "experiments_cross.json")
    return results


# ── 5. Statistical tests ──────────────────────────────────────────────────────

def run_stats(config):
    """Re-runs Wilcoxon tests using saved all_results.json."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Statistical Significance Tests (Wilcoxon)")
    print("=" * 60)

    log_dir  = config["paths"]["logs"]
    res_path = os.path.join(log_dir, "all_results.json")
    if not os.path.exists(res_path):
        print("[SKIP] all_results.json not found. Run main.py first.")
        return {}

    with open(res_path) as f:
        results = json.load(f)

    # Run statistical_tests module directly via subprocess to avoid re-import issues
    import subprocess
    ret = subprocess.run(
        [sys.executable, "-X", "utf8",
         "src/explainability/statistical_tests.py"],
        capture_output=True, text=True
    )
    print(ret.stdout)
    if ret.stderr:
        print("[stderr]", ret.stderr[:500])

    out_path = os.path.join(log_dir, "statistical_tests.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AnomaLens additional experiments")
    parser.add_argument("--exp",
                        choices=["noise", "unseen", "params", "cross", "stats", "all"],
                        default="all")
    args = parser.parse_args()

    config = load_config()
    os.makedirs(config["paths"]["logs"], exist_ok=True)

    all_results = {}

    run_map = {
        "noise"  : run_noise_experiment,
        "unseen" : run_unseen_experiment,
        "params" : run_params_experiment,
        "cross"  : run_cross_dataset_experiment,
        "stats"  : run_stats,
    }

    to_run = list(run_map.keys()) if args.exp == "all" else [args.exp]

    for exp_name in to_run:
        res = run_map[exp_name](config)
        all_results[exp_name] = res

    if args.exp == "all":
        save_json(all_results, config, "experiments_all.json")

    print("\n" + "=" * 60)
    print("All experiments complete. Results in experiments/logs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
