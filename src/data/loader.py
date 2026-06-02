import os
import glob
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_skab(config: dict) -> pd.DataFrame:
    """
    Loads all CSV files from valve1 and valve2 folders,
    concatenates them into a single DataFrame, and adds
    source_group and source_file tracking columns.
    """
    cfg = config["data"]["skab"]
    dfs = []

    for group_key in ["valve1_path", "valve2_path"]:
        folder = cfg[group_key]
        group_name = os.path.basename(folder)  # valve1 or valve2
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {folder}")

        for fpath in csv_files:
            df = pd.read_csv(fpath, sep=";", index_col=False)
            df["source_group"] = group_name
            df["source_file"] = os.path.basename(fpath)
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Validate target column exists
    target = cfg["target_col"]
    if target not in combined.columns:
        raise ValueError(f"Target column '{target}' not found in SKAB data.")

    print(f"[SKAB] Loaded {len(combined)} rows from "
          f"{sum(len(glob.glob(os.path.join(cfg[k], '*.csv'))) for k in ['valve1_path','valve2_path'])} files")
    print(f"[SKAB] Anomaly rate: {combined[target].mean():.2%}")
    print(f"[SKAB] Unique source files: {combined['source_file'].nunique()}")

    return combined


def load_batadal(config: dict) -> pd.DataFrame:
    """
    Loads BATADAL Training Dataset 2.
    Maps ATT_FLAG: -999 -> 0 (normal), 1 -> 1 (attack).
    """
    cfg = config["data"]["batadal"]
    fpath = cfg["path"]

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"BATADAL file not found: {fpath}")

    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()  # remove leading/trailing spaces

    label_col = cfg["label_col"]
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. "
                         f"Available: {df.columns.tolist()}")

    # Map -999 (normal) -> 0, 1 (attack) -> 1
    df[label_col] = df[label_col].map({
        cfg["normal_label"]: 0,
        cfg["attack_label"]: 1
    })

    if df[label_col].isnull().any():
        unexpected = df[label_col].isnull().sum()
        raise ValueError(f"{unexpected} rows have unexpected ATT_FLAG values.")

    print(f"[BATADAL] Loaded {len(df)} rows")
    print(f"[BATADAL] Attack rate: {df[label_col].mean():.2%}")
    print(f"[BATADAL] Date range: {df['DATETIME'].iloc[0]} → {df['DATETIME'].iloc[-1]}")

    return df


def get_skab_features_target(df: pd.DataFrame, config: dict):
    """Returns X (features only), y (target), and groups (for GroupKFold) for SKAB."""
    cfg = config["data"]["skab"]
    drop = cfg["drop_cols"]
    target = cfg["target_col"]

    # Keep source_file for GroupKFold — drop everything else non-feature
    feature_cols = [c for c in df.columns
                    if c not in drop and c != target]

    X = df[feature_cols].copy()
    y = df[target].copy()
    groups = df["source_file"].copy()  # for GroupKFold

    return X, y, groups


def get_batadal_features_target(df: pd.DataFrame, config: dict):
    """Returns X (features only) and y (target) for BATADAL."""
    cfg = config["data"]["batadal"]
    drop = cfg["drop_cols"]
    label = cfg["label_col"]

    feature_cols = [c for c in df.columns
                    if c not in drop and c != label]

    X = df[feature_cols].copy()
    y = df[label].copy()

    return X, y


if __name__ == "__main__":
    config = load_config()

    print("=" * 50)
    skab_df = load_skab(config)
    X_skab, y_skab, groups = get_skab_features_target(skab_df, config)
    print(f"[SKAB] Features shape: {X_skab.shape}")
    print(f"[SKAB] Feature columns: {X_skab.columns.tolist()}")

    print("=" * 50)
    batadal_df = load_batadal(config)
    X_bat, y_bat = get_batadal_features_target(batadal_df, config)
    print(f"[BATADAL] Features shape: {X_bat.shape}")
    print(f"[BATADAL] Feature columns: {X_bat.columns.tolist()}")