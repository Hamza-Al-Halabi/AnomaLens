import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Preprocessor:
    """
    Handles normalization and PCA for both SKAB and BATADAL datasets.
    CRITICAL: fit() must be called on train data only.
    transform() is then applied to val and test sets.
    """

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config["preprocessing"]["pca_components"])
        self._scaler_fitted = False
        self._pca_fitted = False

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """
        Fit scaler and PCA on training data ONLY.
        Never call this on val or test data.
        """
        X_filled = self._fill_missing(X)
        X_scaled = self.scaler.fit_transform(X_filled)
        self._scaler_fitted = True

        self.pca.fit(X_scaled)
        self._pca_fitted = True

        print(f"[Preprocessor] Fitted on {X.shape[0]} rows, {X.shape[1]} features")
        print(f"[Preprocessor] PCA explained variance (PC1): "
              f"{self.pca.explained_variance_ratio_[0]:.2%}")
        return self

    def transform(self, X: pd.DataFrame) -> tuple:
        """
        Returns:
            X_scaled : np.ndarray — normalized features (for deep learning)
            X_pca    : np.ndarray — PC1 only (for automata model)
        """
        if not self._scaler_fitted:
            raise RuntimeError("Call fit() on training data before transform().")

        X_filled = self._fill_missing(X)
        X_scaled = self.scaler.transform(X_filled)
        X_pca = self.pca.transform(X_scaled)  # shape: (n, pca_components)

        return X_scaled, X_pca

    def fit_transform(self, X: pd.DataFrame) -> tuple:
        """Convenience method for training set only."""
        self.fit(X)
        return self.transform(X)

    def _fill_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward fill then backward fill for time series; remaining with 0."""
        X_filled = X.ffill().bfill().fillna(0)
        missing = X.isnull().sum().sum()
        if missing > 0:
            print(f"[Preprocessor] Filled {missing} missing values")
        return X_filled


def split_batadal_temporal(X: pd.DataFrame,
                            y: pd.Series,
                            config: dict) -> tuple:
    """
    Time-ordered split for BATADAL: 60% train, 20% val, 20% test.
    No shuffling — preserves temporal order.
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
    print(f"[BATADAL Split] Attack rate — Train: {y_train.mean():.2%} | "
          f"Val: {y_val.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))

    from src.data.loader import load_config, load_skab, load_batadal
    from src.data.loader import get_skab_features_target, get_batadal_features_target

    config = load_config()

    # ── BATADAL test ──────────────────────────────────────────
    print("=" * 50)
    print("BATADAL preprocessing test")
    batadal_df = load_batadal(config)
    X_bat, y_bat = get_batadal_features_target(batadal_df, config)

    X_train, X_val, X_test, y_train, y_val, y_test = split_batadal_temporal(
        X_bat, y_bat, config
    )

    prep = Preprocessor(config)
    X_train_scaled, X_train_pca = prep.fit_transform(X_train)  # fit on train only
    X_val_scaled,   X_val_pca   = prep.transform(X_val)
    X_test_scaled,  X_test_pca  = prep.transform(X_test)

    print(f"X_train_scaled shape : {X_train_scaled.shape}")
    print(f"X_train_pca shape    : {X_train_pca.shape}  ← PC1 for automata")
    print(f"X_val_scaled shape   : {X_val_scaled.shape}")
    print(f"X_test_scaled shape  : {X_test_scaled.shape}")

    # ── SKAB test (single fold preview) ──────────────────────
    print("=" * 50)
    print("SKAB preprocessing test (single split preview)")
    skab_df = load_skab(config)
    X_skab, y_skab, groups = get_skab_features_target(skab_df, config)

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=config["model"]["skab_n_splits"])
    train_idx, test_idx = next(gkf.split(X_skab, y_skab, groups))

    X_tr = X_skab.iloc[train_idx]
    X_te = X_skab.iloc[test_idx]

    prep_skab = Preprocessor(config)
    X_tr_scaled, X_tr_pca = prep_skab.fit_transform(X_tr)
    X_te_scaled, X_te_pca = prep_skab.transform(X_te)

    print(f"X_train_scaled shape : {X_tr_scaled.shape}")
    print(f"X_train_pca shape    : {X_tr_pca.shape}  ← PC1 for automata")
    print(f"X_test_scaled shape  : {X_te_scaled.shape}")