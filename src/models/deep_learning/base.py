import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, accuracy_score)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset for time series classification.
    Label = majority vote over the window.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = torch.FloatTensor(np.array(X).copy())
        self.y       = torch.FloatTensor(np.array(y).copy())
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        x_window = self.X[idx: idx + self.seq_len]
        y_window = self.y[idx: idx + self.seq_len]
        label    = float(y_window.float().mean() >= 0.5)
        return x_window, torch.tensor(label, dtype=torch.float32)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience   = patience
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Core functions ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / max(len(loader.dataset), 1)


def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    preds    = (np.array(all_probs) >= threshold).astype(int)
    metrics  = compute_metrics(np.array(all_labels), preds)
    return avg_loss, metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred,    zero_division=0)),
        "f1"       : float(f1_score(y_true, y_pred,        zero_division=0)),
    }


# ── Training pipeline ─────────────────────────────────────────────────────────

def train_model(model, config: dict,
                X_train, y_train,
                X_val,   y_val,
                seed: int,
                model_name: str = "model") -> tuple:
    """
    Full training loop with early stopping.
    Uses threshold=0.5 — pos_weight handles class imbalance.
    Returns (result_dict, trained_model).
    """
    set_seed(seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len  = config["model"]["seq_len"]
    batch_sz = config["model"]["batch_size"]
    epochs   = config["model"]["epochs"]
    patience = config["model"]["patience"]
    lr       = config["model"]["learning_rate"]

    train_ds = TimeSeriesDataset(X_train, y_train, seq_len)
    val_ds   = TimeSeriesDataset(X_val,   y_val,   seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False)

    model    = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # pos_weight to handle class imbalance
    pos_count  = float(np.array(y_train).sum())
    neg_count  = float(len(y_train) - pos_count)
    raw_weight = neg_count / max(pos_count, 1)
    max_w = config["model"].get("max_pos_weight", 10.0)
    pos_weight = torch.tensor([min(raw_weight, max_w)]).to(device)
    print(f"  pos_weight={min(raw_weight, max_w):.1f} (raw={raw_weight:.1f})")
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    stopper  = EarlyStopping(patience=patience)
    t_start  = time.time()
    epoch    = 0

    for epoch in range(epochs):
        train_loss            = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_dl, criterion, device)

        if stopper.step(val_loss, model):
            print(f"  [EarlyStopping] epoch {epoch+1}/{epochs} "
                  f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_f1={val_metrics['f1']:.4f}")

    stopper.restore_best(model)
    training_time = time.time() - t_start

    _, val_metrics = evaluate(model, val_dl, criterion, device)

    t_inf = time.time()
    with torch.no_grad():
        for X_batch, _ in val_dl:
            _ = model(X_batch.to(device))
    inference_time = time.time() - t_inf

    result = {
        "model"         : model_name,
        "seed"          : seed,
        "val_metrics"   : val_metrics,
        "training_time" : round(training_time, 2),
        "inference_time": round(inference_time, 4),
        "epochs_run"    : epoch + 1,
    }

    return result, model


def predict_model(model, config: dict,
                  X_test, y_test,
                  model_name: str = "model") -> dict:
    """Run inference on test set. Returns metrics dict."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = config["model"]["seq_len"]
    bs      = config["model"]["batch_size"]

    test_ds   = TimeSeriesDataset(X_test, y_test, seq_len)
    test_dl   = DataLoader(test_ds, batch_size=bs, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    t_inf = time.time()
    _, test_metrics = evaluate(model, test_dl, criterion, device)
    inference_time  = time.time() - t_inf

    test_metrics["inference_time"] = round(inference_time, 4)
    return test_metrics


def save_results(results: dict, config: dict, filename: str):
    """Save experiment results to logs directory."""
    log_dir = config["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Logger] Saved → {path}")