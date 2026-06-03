import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    1D-CNN binary classifier for time series anomaly detection.
    Input : (batch, seq_len, n_features)
    Output: (batch, 1) — raw logit
    """

    def __init__(self, config: dict, n_features: int):
        super().__init__()
        hidden = config["model"]["hidden_size"]
        drop   = config["model"]["dropout"]
        seq_len = config["model"]["seq_len"]

        # Conv layers expect (batch, channels, seq_len)
        # n_features treated as channels
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, hidden,     kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Conv1d(hidden,     hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Conv1d(hidden * 2, hidden,     kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)   # global average pooling → (batch, hidden, 1)
        self.dropout = nn.Dropout(drop)
        self.fc      = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)      # → (batch, n_features, seq_len)
        x = self.conv_block(x)      # → (batch, hidden, seq_len)
        x = self.pool(x)            # → (batch, hidden, 1)
        x = x.squeeze(-1)           # → (batch, hidden)
        x = self.dropout(x)
        return self.fc(x)           # → (batch, 1)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))

    from src.data.loader       import load_config, load_batadal, get_batadal_features_target
    from src.data.splitter     import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.deep_learning.base import train_model, predict_model

    config = load_config()
    seeds  = config["model"]["seeds"]

    print("=" * 55)
    print("1D-CNN — BATADAL quick test (1 seed)")

    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    X_tr_sc, _ = prep.fit_transform(X_train)
    X_va_sc, _ = prep.transform(X_val)
    X_te_sc, _ = prep.transform(X_test)

    n_features = X_tr_sc.shape[1]
    model = CNN1DModel(config, n_features)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    result, trained_model = train_model(
        model, config,
        X_tr_sc, y_train.values,
        X_va_sc, y_val.values,
        seed=seeds[0], model_name="1D-CNN"
    )

    test_metrics = predict_model(trained_model, config,
                                  X_te_sc, y_test.values,
                                  model_name="1D-CNN")

    print(f"\nVal  metrics : {result['val_metrics']}")
    print(f"Test metrics : {test_metrics}")
    print(f"Training time: {result['training_time']}s")