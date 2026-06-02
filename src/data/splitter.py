import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def get_skab_folds(X: pd.DataFrame,
                   y: pd.Series,
                   groups: pd.Series,
                   config: dict) -> list:
    """
    Returns list of (train_idx, test_idx) tuples using GroupKFold.
    Groups are source_file — no file appears in both train and test.

    Uses StratifiedGroupKFold if class balance allows, else falls back
    to GroupKFold.
    """
    n_splits = config["model"]["skab_n_splits"]

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        folds = list(sgkf.split(X, y, groups))
        print(f"[Splitter] Using StratifiedGroupKFold — {n_splits} folds")
    except Exception:
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(X, y, groups))
        print(f"[Splitter] Using GroupKFold (fallback) — {n_splits} folds")

    # Print fold summary
    for i, (tr_idx, te_idx) in enumerate(folds):
        tr_files = groups.iloc[tr_idx].unique()
        te_files = groups.iloc[te_idx].unique()
        overlap  = set(tr_files) & set(te_files)
        anomaly_rate = y.iloc[te_idx].mean()
        print(f"  Fold {i+1}: train={len(tr_idx):>6} | test={len(te_idx):>5} | "
              f"test_anomaly={anomaly_rate:.2%} | "
              f"test_files={te_files.tolist()} | "
              f"overlap={'⚠️ YES' if overlap else 'OK'}")

    return folds


def get_batadal_split(X: pd.DataFrame,
                      y: pd.Series,
                      config: dict) -> tuple:
    """
    Time-ordered split: 60% train / 20% val / 20% test.
    No shuffling — preserves temporal order.
    Imported here for convenience; actual logic lives in preprocessor.py.
    """
    split_cfg = config["data"]["batadal"]["split"]
    n = len(X)
    train_end = int(n * split_cfg["train"])
    val_end   = int(n * (split_cfg["train"] + split_cfg["val"]))

    X_train = X.iloc[:train_end].reset_index(drop=True)
    X_val   = X.iloc[train_end:val_end].reset_index(drop=True)
    X_test  = X.iloc[val_end:].reset_index(drop=True)

    y_train = y.iloc[:train_end].reset_index(drop=True)
    y_val   = y.iloc[train_end:val_end].reset_index(drop=True)
    y_test  = y.iloc[val_end:].reset_index(drop=True)

    print(f"[BATADAL Split] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"[BATADAL Split] Attack rate — "
          f"Train: {y_train.mean():.2%} | "
          f"Val: {y_val.mean():.2%} | "
          f"Test: {y_test.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))

    from src.data.loader import (load_config, load_skab, load_batadal,
                                  get_skab_features_target,
                                  get_batadal_features_target)

    config = load_config()

    # ── SKAB folds ────────────────────────────────────────────
    print("=" * 55)
    print("SKAB — GroupKFold split validation")
    skab_df = load_skab(config)
    X_skab, y_skab, groups = get_skab_features_target(skab_df, config)
    folds = get_skab_folds(X_skab, y_skab, groups, config)
    print(f"\nTotal folds: {len(folds)} ✓")

    # ── BATADAL temporal split ────────────────────────────────
    print("=" * 55)
    print("BATADAL — Temporal split validation")
    batadal_df = load_batadal(config)
    X_bat, y_bat = get_batadal_features_target(batadal_df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(
        X_bat, y_bat, config
    )
    print(f"\nSplit ratios: "
          f"{len(X_train)/len(X_bat):.0%} / "
          f"{len(X_val)/len(X_bat):.0%} / "
          f"{len(X_test)/len(X_bat):.0%} ✓")