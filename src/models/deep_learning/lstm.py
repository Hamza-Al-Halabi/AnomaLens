import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based binary classifier for time series anomaly detection.
    Input : (batch, seq_len, n_features)
    Output: (batch, 1) — raw logit
    """

    def __init__(self, config: dict, n_features: int):
        super().__init__()
        hidden = config["model"]["hidden_size"]
        layers = config["model"]["num_layers"]
        drop   = config["model"]["dropout"]

        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden,
            num_layers   = layers,
            batch_first  = True,
            dropout      = drop if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(drop)
        self.fc      = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)        # (batch, seq_len, hidden)
        out    = out[:, -1, :]       # last timestep: (batch, hidden)
        out    = self.dropout(out)
        return self.fc(out)          # (batch, 1)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))

    import numpy as np
    from src.data.loader      import load_config, load_batadal, get_batadal_features_target
    from src.data.splitter    import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.deep_learning.base import train_model, predict_model

    config  = load_config()
    seeds   = config["model"]["seeds"]

    print("=" * 55)
    print("LSTM — BATADAL quick test (1 seed)")

    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    X_tr_sc, _ = prep.fit_transform(X_train)
    X_va_sc, _ = prep.transform(X_val)
    X_te_sc, _ = prep.transform(X_test)

    n_features = X_tr_sc.shape[1]
    model = LSTMModel(config, n_features)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    result, trained_model = train_model(
        model, config,
        X_tr_sc, y_train.values,
        X_va_sc, y_val.values,
        seed=seeds[0], model_name="LSTM"
    )

    test_metrics = predict_model(trained_model, config,
                                  X_te_sc, y_test.values,
                                  model_name="LSTM")

    print(f"\nVal  metrics : {result['val_metrics']}")
    print(f"Test metrics : {test_metrics}")
    print(f"Training time: {result['training_time']}s")