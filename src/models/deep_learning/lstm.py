import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, config: dict, n_features: int):
        super().__init__()
        hidden = config["model"]["hidden_size"]
        layers = config["model"]["num_layers"]
        drop   = config["model"]["dropout"]
        self.lstm    = nn.LSTM(n_features, hidden, layers,
                               batch_first=True,
                               dropout=drop if layers > 1 else 0.0)
        self.dropout = nn.Dropout(drop)
        self.fc      = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from src.data.loader       import load_config, load_batadal, get_batadal_features_target
    from src.data.splitter     import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.deep_learning.base import train_model, predict_model

    config = load_config()
    print("=" * 55)
    print("LSTM — BATADAL quick test (seed=42)")

    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    X_tr_sc, _ = prep.fit_transform(X_train)
    X_va_sc, _ = prep.transform(X_val)
    X_te_sc, _ = prep.transform(X_test)

    model = LSTMModel(config, X_tr_sc.shape[1])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    result, trained = train_model(model, config,
                                   X_tr_sc, y_train.values,
                                   X_va_sc, y_val.values,
                                   seed=42, model_name="LSTM")
    test_m = predict_model(trained, config, X_te_sc, y_test.values, model_name="LSTM")

    print(f"\nVal  : {result['val_metrics']}")
    print(f"Test : {test_m}")
    print(f"Time : {result['training_time']}s")