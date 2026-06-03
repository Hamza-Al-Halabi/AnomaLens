"""
AnomaLens — Main Pipeline
Runs all models both datasets all seeds and saves results to JSON.
Usage:
    python main.py                    # full run
    python main.py --dataset batadal  # single dataset
    python main.py --model gru        # single model
"""

import os
import sys
import json
import argparse
import numpy as np

# Force UTF-8 stdout/stderr so Unicode characters print on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Make sure src is importable
sys.path.insert(0, os.path.abspath("."))

from src.data.loader       import (load_config, load_skab, load_batadal,
                                    get_skab_features_target,
                                    get_batadal_features_target)
from src.data.splitter     import get_skab_folds, get_batadal_split
from src.data.preprocessor import Preprocessor
from src.models.deep_learning.base  import train_model, predict_model, save_results
from src.models.deep_learning.lstm  import LSTMModel
from src.models.deep_learning.gru   import GRUModel
from src.models.deep_learning.cnn1d import CNN1DModel
from src.models.automata.automata_model import ProbabilisticAutomata
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_std(values: list) -> tuple:
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred,    zero_division=0)),
        "f1"       : float(f1_score(y_true, y_pred,        zero_division=0)),
    }


def make_dl_model(name: str, config: dict, n_features: int):
    name = name.lower()
    if name == "lstm":
        return LSTMModel(config, n_features)
    elif name == "gru":
        return GRUModel(config, n_features)
    elif name in ("cnn", "1d-cnn", "cnn1d"):
        return CNN1DModel(config, n_features)
    raise ValueError(f"Unknown model: {name}")


# ── BATADAL pipeline ──────────────────────────────────────────────────────────

def run_batadal(config: dict, models: list) -> dict:
    print("\n" + "=" * 60)
    print("DATASET: BATADAL")
    print("=" * 60)

    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    seeds   = config["model"]["seeds"]
    results = {}

    # ── Deep learning models ──────────────────────────────────
    for model_name in [m for m in models if m != "automata"]:
        print(f"\n── {model_name.upper()} on BATADAL ──")
        seed_f1s, seed_prec, seed_rec, seed_acc = [], [], [], []
        train_times, inf_times = [], []

        for seed in seeds:
            print(f"  seed={seed}")
            prep = Preprocessor(config)
            X_tr_sc, _ = prep.fit_transform(X_train)
            X_va_sc, _ = prep.transform(X_val)
            X_te_sc, _ = prep.transform(X_test)

            model = make_dl_model(model_name, config, X_tr_sc.shape[1])
            result, trained = train_model(
                model, config,
                X_tr_sc, y_train.values,
                X_va_sc, y_val.values,
                seed=seed, model_name=model_name
            )
            test_m = predict_model(trained, config, X_te_sc, y_test.values)

            seed_f1s.append(test_m["f1"])
            seed_prec.append(test_m["precision"])
            seed_rec.append(test_m["recall"])
            seed_acc.append(test_m["accuracy"])
            train_times.append(result["training_time"])
            inf_times.append(test_m["inference_time"])

        f1_mean, f1_std = mean_std(seed_f1s)
        results[model_name] = {
            "dataset"       : "BATADAL",
            "f1_mean"       : f1_mean,
            "f1_std"        : f1_std,
            "precision_mean": mean_std(seed_prec)[0],
            "recall_mean"   : mean_std(seed_rec)[0],
            "accuracy_mean" : mean_std(seed_acc)[0],
            "train_time_mean": mean_std(train_times)[0],
            "inf_time_mean" : mean_std(inf_times)[0],
            "per_seed_f1"   : seed_f1s,
        }
        print(f"  ✓ {model_name} BATADAL: F1={f1_mean:.4f} ± {f1_std:.4f}")

    # ── Automata model ────────────────────────────────────────
    if "automata" in models:
        print(f"\n── AUTOMATA on BATADAL ──")
        seed_f1s, seed_prec, seed_rec, seed_acc = [], [], [], []

        for seed in seeds:
            np.random.seed(seed)
            prep = Preprocessor(config)
            _, X_tr_pca = prep.fit_transform(X_train)
            _, X_va_pca = prep.transform(X_val)
            _, X_te_pca = prep.transform(X_test)

            automata = ProbabilisticAutomata(config)
            automata.fit(X_tr_pca[:, 0])
            automata.calibrate_threshold(X_va_pca[:, 0], y_val.values)
            y_pred, _ = automata.predict(X_te_pca[:, 0])

            min_len = min(len(y_test), len(y_pred))
            m = compute_metrics(y_test.values[:min_len], y_pred[:min_len])
            seed_f1s.append(m["f1"])
            seed_prec.append(m["precision"])
            seed_rec.append(m["recall"])
            seed_acc.append(m["accuracy"])
            print(f"  seed={seed} F1={m['f1']:.4f}")

        f1_mean, f1_std = mean_std(seed_f1s)
        results["automata"] = {
            "dataset"       : "BATADAL",
            "f1_mean"       : f1_mean,
            "f1_std"        : f1_std,
            "precision_mean": mean_std(seed_prec)[0],
            "recall_mean"   : mean_std(seed_rec)[0],
            "accuracy_mean" : mean_std(seed_acc)[0],
            "per_seed_f1"   : seed_f1s,
        }
        print(f"  ✓ Automata BATADAL: F1={f1_mean:.4f} ± {f1_std:.4f}")

    return results


# ── SKAB pipeline ─────────────────────────────────────────────────────────────

def run_skab(config: dict, models: list) -> dict:
    print("\n" + "=" * 60)
    print("DATASET: SKAB")
    print("=" * 60)

    df = load_skab(config)
    X, y, groups = get_skab_features_target(df, config)
    folds  = get_skab_folds(X, y, groups, config)
    seeds  = config["model"]["seeds"]
    results = {}

    # ── Deep learning models ──────────────────────────────────
    for model_name in [m for m in models if m != "automata"]:
        print(f"\n── {model_name.upper()} on SKAB ──")
        all_f1s, all_prec, all_rec, all_acc = [], [], [], []
        train_times, inf_times = [], []

        for fold_i, (tr_idx, te_idx) in enumerate(folds):
            X_tr_raw = X.iloc[tr_idx]
            X_te_raw = X.iloc[te_idx]
            y_tr     = y.iloc[tr_idx]
            y_te     = y.iloc[te_idx]

            # Use first 80% of train fold as train, last 20% as val
            n_val    = int(len(X_tr_raw) * 0.2)
            X_val_raw = X_tr_raw.iloc[-n_val:]
            X_tr_raw  = X_tr_raw.iloc[:-n_val]
            y_val     = y_tr.iloc[-n_val:]
            y_tr      = y_tr.iloc[:-n_val]

            fold_f1s = []
            for seed in seeds:
                prep = Preprocessor(config)
                X_tr_sc, _ = prep.fit_transform(X_tr_raw)
                X_va_sc, _ = prep.transform(X_val_raw)
                X_te_sc, _ = prep.transform(X_te_raw)

                model  = make_dl_model(model_name, config, X_tr_sc.shape[1])
                result, trained = train_model(
                    model, config,
                    X_tr_sc, y_tr.values,
                    X_va_sc, y_val.values,
                    seed=seed, model_name=model_name
                )
                test_m = predict_model(trained, config, X_te_sc, y_te.values)

                fold_f1s.append(test_m["f1"])
                all_f1s.append(test_m["f1"])
                all_prec.append(test_m["precision"])
                all_rec.append(test_m["recall"])
                all_acc.append(test_m["accuracy"])
                train_times.append(result["training_time"])
                inf_times.append(test_m["inference_time"])

            fold_mean = float(np.mean(fold_f1s))
            print(f"  Fold {fold_i+1}: F1={fold_mean:.4f} "
                  f"(seeds: {[round(f,3) for f in fold_f1s]})")

        f1_mean, f1_std = mean_std(all_f1s)
        results[model_name] = {
            "dataset"       : "SKAB",
            "f1_mean"       : f1_mean,
            "f1_std"        : f1_std,
            "precision_mean": mean_std(all_prec)[0],
            "recall_mean"   : mean_std(all_rec)[0],
            "accuracy_mean" : mean_std(all_acc)[0],
            "train_time_mean": mean_std(train_times)[0],
            "inf_time_mean" : mean_std(inf_times)[0],
            "per_seed_f1"   : all_f1s,
        }
        print(f"  ✓ {model_name} SKAB: F1={f1_mean:.4f} ± {f1_std:.4f}")

    # ── Automata model ────────────────────────────────────────
    if "automata" in models:
        print(f"\n── AUTOMATA on SKAB ──")
        all_f1s, all_prec, all_rec, all_acc = [], [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(folds):
            X_tr_raw = X.iloc[tr_idx]
            X_te_raw = X.iloc[te_idx]
            y_tr     = y.iloc[tr_idx]
            y_te     = y.iloc[te_idx]

            n_val     = int(len(X_tr_raw) * 0.2)
            X_val_raw = X_tr_raw.iloc[-n_val:]
            X_tr_raw  = X_tr_raw.iloc[:-n_val]
            y_val_s   = y_tr.iloc[-n_val:]
            y_tr      = y_tr.iloc[:-n_val]

            fold_f1s = []
            for seed in seeds:
                np.random.seed(seed)
                prep = Preprocessor(config)
                _, X_tr_pca = prep.fit_transform(X_tr_raw)
                _, X_va_pca = prep.transform(X_val_raw)
                _, X_te_pca = prep.transform(X_te_raw)

                automata = ProbabilisticAutomata(config)
                automata.fit(X_tr_pca[:, 0])
                automata.calibrate_threshold(X_va_pca[:, 0], y_val_s.values)
                y_pred, _ = automata.predict(X_te_pca[:, 0])

                min_len = min(len(y_te), len(y_pred))
                m = compute_metrics(y_te.values[:min_len], y_pred[:min_len])
                fold_f1s.append(m["f1"])
                all_f1s.append(m["f1"])
                all_prec.append(m["precision"])
                all_rec.append(m["recall"])
                all_acc.append(m["accuracy"])

            print(f"  Fold {fold_i+1}: F1={float(np.mean(fold_f1s)):.4f}")

        f1_mean, f1_std = mean_std(all_f1s)
        results["automata"] = {
            "dataset"       : "SKAB",
            "f1_mean"       : f1_mean,
            "f1_std"        : f1_std,
            "precision_mean": mean_std(all_prec)[0],
            "recall_mean"   : mean_std(all_rec)[0],
            "accuracy_mean" : mean_std(all_acc)[0],
            "per_seed_f1"   : all_f1s,
        }
        print(f"  ✓ Automata SKAB: F1={f1_mean:.4f} ± {f1_std:.4f}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AnomaLens pipeline")
    parser.add_argument("--dataset", choices=["batadal", "skab", "all"],
                        default="all")
    parser.add_argument("--model",
                        choices=["lstm", "gru", "cnn", "automata", "all"],
                        default="all")
    args = parser.parse_args()

    config  = load_config()
    os.makedirs(config["paths"]["logs"], exist_ok=True)

    all_models = ["lstm", "gru", "cnn", "automata"]
    models = all_models if args.model == "all" else [args.model]
    datasets = ["batadal", "skab"] if args.dataset == "all" else [args.dataset]

    all_results = {}

    if "batadal" in datasets:
        batadal_res = run_batadal(config, models)
        all_results["BATADAL"] = batadal_res
        save_results(batadal_res, config, "batadal_results.json")

    if "skab" in datasets:
        skab_res = run_skab(config, models)
        all_results["SKAB"] = skab_res
        save_results(skab_res, config, "skab_results.json")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY — Test F1 (mean ± std across seeds)")
    print("=" * 60)
    print(f"{'Model':<12} {'BATADAL':>20} {'SKAB':>20}")
    print("-" * 55)
    for m in all_models:
        bat = all_results.get("BATADAL", {}).get(m, {})
        skb = all_results.get("SKAB", {}).get(m, {})
        bat_str = f"{bat.get('f1_mean',0):.4f}±{bat.get('f1_std',0):.4f}" if bat else "—"
        skb_str = f"{skb.get('f1_mean',0):.4f}±{skb.get('f1_std',0):.4f}" if skb else "—"
        print(f"{m:<12} {bat_str:>20} {skb_str:>20}")

    save_results(all_results, config, "all_results.json")
    print("\nAll results saved to experiments/logs/")


if __name__ == "__main__":
    main()