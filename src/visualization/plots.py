"""
Visualization module — generates all figures required by the project:
  - Confusion matrices (per model)
  - Precision-Recall curves
  - Automata state diagram
  - Transition probability heatmap
  - Parameter sensitivity plots (window_size, alphabet_size)
  - F1 comparison bar charts (per dataset)

All figures saved to results/figures/.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             ConfusionMatrixDisplay)

sys.path.insert(0, os.path.abspath("."))


def ensure_dir(config):
    d = config["paths"]["figures"]
    os.makedirs(d, exist_ok=True)
    return d


# ── 1. F1 comparison bar chart ────────────────────────────────────────────────

def plot_f1_comparison(config):
    fig_dir = ensure_dir(config)
    with open(os.path.join(config["paths"]["logs"], "all_results.json")) as f:
        results = json.load(f)

    models   = ["lstm", "gru", "cnn", "automata"]
    datasets = ["SKAB", "BATADAL"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, datasets):
        means = [results[ds][m]["f1_mean"] for m in models]
        stds  = [results[ds][m]["f1_std"]  for m in models]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
        bars = ax.bar([m.upper() for m in models], means, yerr=stds,
                      capsize=5, color=colors, alpha=0.85)
        ax.set_title(f"{ds} — Test F1 (mean ± std, 5 seeds)", fontsize=12)
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.02,
                    f"{mean:.3f}", ha="center", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 2. Confusion matrices ──────────────────────────────────────────────────────

def plot_confusion_matrices(config, dataset="BATADAL"):
    from src.data.loader       import (load_config, load_batadal,
                                        get_batadal_features_target)
    from src.data.splitter     import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.deep_learning.base  import train_model, TimeSeriesDataset
    from src.models.deep_learning.lstm  import LSTMModel
    from src.models.deep_learning.gru   import GRUModel
    from src.models.deep_learning.cnn1d import CNN1DModel
    from src.models.automata.automata_model import ProbabilisticAutomata
    from torch.utils.data import DataLoader
    import torch

    fig_dir = ensure_dir(config)
    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)
    seed = config["model"]["seeds"][0]

    prep = Preprocessor(config)
    X_tr_sc, X_tr_pca = prep.fit_transform(X_train)
    X_va_sc, X_va_pca = prep.transform(X_val)
    X_te_sc, X_te_pca = prep.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = config["model"]["seq_len"]

    preds_dict = {}

    # DL models
    for name, Cls in [("LSTM", LSTMModel), ("GRU", GRUModel), ("1D-CNN", CNN1DModel)]:
        model = Cls(config, X_tr_sc.shape[1])
        _, trained = train_model(model, config, X_tr_sc, y_train.values,
                                  X_va_sc, y_val.values, seed=seed, model_name=name)
        thr = getattr(trained, "threshold_", 0.5)
        test_ds = TimeSeriesDataset(X_te_sc, y_test.values, seq_len)
        test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
        trained.eval()
        probs, labels = [], []
        with torch.no_grad():
            for Xb, yb in test_dl:
                probs.extend(torch.sigmoid(trained(Xb.to(device)).squeeze(-1)).cpu().numpy())
                labels.extend(yb.numpy())
        preds_dict[name] = (np.array(labels), (np.array(probs) >= thr).astype(int))

    # Automata
    automata = ProbabilisticAutomata(config)
    automata.fit(X_tr_pca[:, 0])
    automata.calibrate_threshold(X_va_pca[:, 0], y_val.values)
    y_pred, _ = automata.predict(X_te_pca[:, 0])
    ml = min(len(y_test), len(y_pred))
    preds_dict["Automata"] = (y_test.values[:ml], y_pred[:ml])

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, (yt, yp)) in zip(axes.flat, preds_dict.items()):
        cm = confusion_matrix(yt, yp)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} — {dataset}")

    plt.tight_layout()
    path = os.path.join(fig_dir, f"confusion_matrices_{dataset.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 3. Automata state diagram ──────────────────────────────────────────────────

def plot_state_diagram(config, top_n=15):
    from src.data.loader       import (load_batadal, get_batadal_features_target)
    from src.data.splitter     import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.automata.automata_model import ProbabilisticAutomata

    fig_dir = ensure_dir(config)
    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    _, X_tr_pca = prep.fit_transform(X_train)
    automata = ProbabilisticAutomata(config)
    automata.fit(X_tr_pca[:, 0])

    # Build graph from top transitions by probability
    G = nx.DiGraph()
    edges = []
    for src, dst_probs in automata.transition_probs.items():
        for dst, prob in dst_probs.items():
            edges.append((src, dst, prob))
    edges.sort(key=lambda e: e[2], reverse=True)
    edges = edges[:top_n]

    for src, dst, prob in edges:
        G.add_edge(src, dst, weight=prob)

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42, k=0.8)
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=1400,
                           node_color="#8172B3", alpha=0.85)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white",
                            font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=[w*4 for w in weights],
                           edge_color="#555", alpha=0.6,
                           arrowsize=18, connectionstyle="arc3,rad=0.1")
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(f"Automata State Diagram (top {top_n} transitions) — BATADAL",
              fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(fig_dir, "automata_state_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 4. Transition probability heatmap ──────────────────────────────────────────

def plot_transition_heatmap(config, top_n=20):
    from src.data.loader       import (load_batadal, get_batadal_features_target)
    from src.data.splitter     import get_batadal_split
    from src.data.preprocessor import Preprocessor
    from src.models.automata.automata_model import ProbabilisticAutomata

    fig_dir = ensure_dir(config)
    df = load_batadal(config)
    X, y = get_batadal_features_target(df, config)
    X_train, X_val, X_test, y_train, y_val, y_test = get_batadal_split(X, y, config)

    prep = Preprocessor(config)
    _, X_tr_pca = prep.fit_transform(X_train)
    automata = ProbabilisticAutomata(config)
    automata.fit(X_tr_pca[:, 0])

    matrix = automata.get_transition_matrix_df()

    # Keep most active states
    activity = matrix.sum(axis=1).sort_values(ascending=False)
    top_states = activity.head(top_n).index.tolist()
    sub = matrix.loc[top_states, top_states]

    plt.figure(figsize=(11, 9))
    sns.heatmap(sub, cmap="viridis", square=True,
                cbar_kws={"label": "Transition probability"})
    plt.title(f"Transition Probability Heatmap (top {top_n} states) — BATADAL",
              fontsize=12)
    plt.xlabel("To state")
    plt.ylabel("From state")
    plt.tight_layout()
    path = os.path.join(fig_dir, "transition_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 5. Parameter sensitivity plots ─────────────────────────────────────────────

def plot_parameter_sensitivity(config):
    fig_dir = ensure_dir(config)
    exp_path = os.path.join(config["paths"]["logs"], "experiments_params.json")
    if not os.path.exists(exp_path):
        exp_path = os.path.join(config["paths"]["logs"], "experiments_all.json")
    if not os.path.exists(exp_path):
        print("  [skip] No parameter experiment results found. "
              "Run: python experiments/run_experiments.py --exp params")
        return

    with open(exp_path) as f:
        data = json.load(f)
    params = data.get("params", data)

    ws_data = params["window_size"]
    al_data = params["alphabet_size"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Window size
    ws_keys = sorted(ws_data.keys(), key=int)
    ws_f1   = [ws_data[k]["f1"] for k in ws_keys]
    ws_states = [ws_data[k]["n_states"] for k in ws_keys]

    ax1 = axes[0]
    ax1.plot(ws_keys, ws_f1, "o-", color="#4C72B0", label="F1", linewidth=2)
    ax1.set_xlabel("Window size")
    ax1.set_ylabel("F1-score", color="#4C72B0")
    ax1.set_title("Window Size Sensitivity (alphabet=3)")
    ax1b = ax1.twinx()
    ax1b.plot(ws_keys, ws_states, "s--", color="#C44E52", label="States", alpha=0.7)
    ax1b.set_ylabel("Number of states", color="#C44E52")
    ax1.grid(alpha=0.3)

    # Alphabet size
    al_keys = sorted(al_data.keys(), key=int)
    al_f1   = [al_data[k]["f1"] for k in al_keys]
    al_states = [al_data[k]["n_states"] for k in al_keys]

    ax2 = axes[1]
    ax2.plot(al_keys, al_f1, "o-", color="#55A868", label="F1", linewidth=2)
    ax2.set_xlabel("Alphabet size")
    ax2.set_ylabel("F1-score", color="#55A868")
    ax2.set_title("Alphabet Size Sensitivity (window=4)")
    ax2b = ax2.twinx()
    ax2b.plot(al_keys, al_states, "s--", color="#C44E52", label="States", alpha=0.7)
    ax2b.set_ylabel("Number of states", color="#C44E52")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "parameter_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


if __name__ == "__main__":
    from src.data.loader import load_config
    config = load_config()

    print("=" * 55)
    print("Generating all figures...")
    print("=" * 55)

    print("\n1. F1 comparison bar chart")
    plot_f1_comparison(config)

    print("\n2. Confusion matrices (BATADAL)")
    plot_confusion_matrices(config, "BATADAL")

    print("\n3. Automata state diagram")
    plot_state_diagram(config)

    print("\n4. Transition probability heatmap")
    plot_transition_heatmap(config)

    print("\n5. Parameter sensitivity plots")
    plot_parameter_sensitivity(config)

    print("\n✓ All figures saved to results/figures/")