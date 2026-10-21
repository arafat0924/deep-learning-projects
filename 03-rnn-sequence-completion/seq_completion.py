"""Multi-target outputs neural network for sequence completion (Chapter 7).

Task: every sequence has six items. The first two items (letters A-E) uniquely
determine the remaining four (digits 1-3). The network reads the sequence one
item per time step and, at every step, predicts the next item (multi-target
outputs). At test time only the first two items are given; the network then
completes the sequence by feeding each prediction back as the next input.

Everything (forward pass, backpropagation for multi-target outputs, weight
updates) is written by hand in NumPy, following the BP algorithm of Chapter 7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TRAIN = [
    "AA1212", "AC1231", "AD1221", "AE1213",
    "BA2312", "BB2323", "BC2331", "BE2313",
    "CB3123", "CC3131", "CD3121", "CE3113",
    "DA2112", "DB2123", "DC2131", "DD2121",
    "EA1312", "EB1323", "ED1321", "EE1313",
]
TEST = ["AB1223", "BD2321", "CA3112", "DE2113", "EC1331"]

SYMBOLS = ["A", "B", "C", "D", "E", "1", "2", "3"]
IDX = {c: i for i, c in enumerate(SYMBOLS)}
K = len(SYMBOLS)          # one-hot size of an item (external input / output)
T = 5                     # time steps: inputs are items 1..5, targets items 2..6
# Item 2 is free (not determined by item 1), so the output at t=1 has no
# target. Outputs at t=2..5 (items 3..6) are the four targets.
TARGET_MASK = np.array([0.0, 1.0, 1.0, 1.0, 1.0])


def encode(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return one-hot inputs X[N,T,K] and integer targets Y[N,T]."""
    X = np.zeros((len(seqs), T, K))
    Y = np.zeros((len(seqs), T), dtype=int)
    for n, s in enumerate(seqs):
        for t in range(T):
            X[n, t, IDX[s[t]]] = 1.0
            Y[n, t] = IDX[s[t + 1]]
    return X, Y


# ---------------------------------------------------------------------------
# Network: external input x_t and internal input h_{t-1} are combined in the
# first layer (slide 13), output layer gives one target per time step.
#   a_t = W x_t + U h_{t-1} + b      h_t = sigmoid(a_t)
#   z_t = V h_t + c                  y_t = softmax(z_t)
# Cost  E = (1/N) sum_n sum_t m_t * CE(y_t, d_t)
# ---------------------------------------------------------------------------

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def init_params(hidden: int, rng: np.random.Generator) -> dict:
    s_in, s_h = 1.0 / np.sqrt(K), 1.0 / np.sqrt(hidden)
    return {
        "W": rng.normal(0, s_in, (hidden, K)),
        "U": rng.normal(0, s_h, (hidden, hidden)),
        "b": np.zeros(hidden),
        "V": rng.normal(0, s_h, (K, hidden)),
        "c": np.zeros(K),
    }


def forward(p: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced forward pass. Returns hidden H[N,T+1,h] (H[:,0]=0) and outputs Yhat[N,T,K]."""
    N = X.shape[0]
    hidden = p["b"].shape[0]
    H = np.zeros((N, T + 1, hidden))
    Yhat = np.zeros((N, T, K))
    for t in range(T):
        H[:, t + 1] = sigmoid(X[:, t] @ p["W"].T + H[:, t] @ p["U"].T + p["b"])
        Yhat[:, t] = softmax(H[:, t + 1] @ p["V"].T + p["c"])
    return H, Yhat


def loss_fn(Yhat: np.ndarray, Y: np.ndarray, p: dict, l2: float) -> float:
    N = Y.shape[0]
    probs = Yhat[np.arange(N)[:, None], np.arange(T)[None, :], Y]
    ce = -(TARGET_MASK * np.log(probs + 1e-12)).sum() / N
    return ce + 0.5 * l2 * (np.sum(p["W"] ** 2) + np.sum(p["U"] ** 2) + np.sum(p["V"] ** 2))


def backward(p: dict, X: np.ndarray, Y: np.ndarray, H: np.ndarray, Yhat: np.ndarray, l2: float) -> dict:
    """BP for multi-target outputs (backpropagation through time).

    delta_t (hidden) = explicit part from output t  +  implicit part through
    later outputs, carried back by U:  delta_t = (V^T e_t + U^T delta_{t+1}) * h_t(1-h_t)
    """
    N = X.shape[0]
    D = np.zeros_like(Yhat)
    D[np.arange(N)[:, None], np.arange(T)[None, :], Y] = 1.0
    g = {k: np.zeros_like(v) for k, v in p.items()}
    delta_next = np.zeros((N, p["b"].shape[0]))
    for t in reversed(range(T)):
        e_t = TARGET_MASK[t] * (Yhat[:, t] - D[:, t]) / N        # output-layer error (softmax + CE)
        g["V"] += e_t.T @ H[:, t + 1]
        g["c"] += e_t.sum(0)
        h = H[:, t + 1]
        delta = (e_t @ p["V"] + delta_next @ p["U"]) * h * (1 - h)  # explicit + implicit
        g["W"] += delta.T @ X[:, t]
        g["U"] += delta.T @ H[:, t]
        g["b"] += delta.sum(0)
        delta_next = delta
    for k in ("W", "U", "V"):
        g[k] += l2 * p[k]
    return g


def gradient_check(seed: int = 0, hidden: int = 6, l2: float = 1e-3) -> float:
    """Compare analytic BP gradients with central finite differences."""
    rng = np.random.default_rng(seed)
    p = init_params(hidden, rng)
    X, Y = encode(TRAIN)
    H, Yhat = forward(p, X)
    g = backward(p, X, Y, H, Yhat, l2)
    eps, worst = 1e-5, 0.0
    for k in p:
        it = np.nditer(p[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[k][i]
            p[k][i] = old + eps
            lp = loss_fn(forward(p, X)[1], Y, p, l2)
            p[k][i] = old - eps
            lm = loss_fn(forward(p, X)[1], Y, p, l2)
            p[k][i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - g[k][i]) / max(1e-8, abs(num) + abs(g[k][i]))
            worst = max(worst, rel)
    return worst


# ---------------------------------------------------------------------------
# Inference: complete a sequence from its first two items
# ---------------------------------------------------------------------------

def complete(p: dict, prefix: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Free-running generation. Returns completed sequence, output probabilities
    for items 3..6 (shape [4,K]) and hidden states h_1..h_5."""
    hidden = p["b"].shape[0]
    h = np.zeros(hidden)
    seq = list(prefix)
    probs, hs = [], []
    for t in range(T):
        x = np.zeros(K)
        x[IDX[seq[t]]] = 1.0
        h = sigmoid(p["W"] @ x + p["U"] @ h + p["b"])
        y = softmax(p["V"] @ h + p["c"])
        hs.append(h)
        if t >= 1:                      # outputs at t=2..5 are the targets (items 3..6)
            probs.append(y)
            if len(seq) < 6:
                seq.append(SYMBOLS[int(np.argmax(y))])
    return "".join(seq), np.array(probs), np.array(hs)


def evaluate(p: dict, seqs: list[str]) -> dict:
    rows, item_ok, seq_ok = [], 0, 0
    for s in seqs:
        pred, probs, _ = complete(p, s[:2])
        ok = [pred[i] == s[i] for i in range(2, 6)]
        item_ok += sum(ok)
        seq_ok += all(ok)
        rows.append({
            "input": s[:2], "target": s, "prediction": pred, "correct": bool(all(ok)),
            "confidence": [float(probs[i, IDX[pred[i + 2]]]) for i in range(4)],
            "p_target": [float(probs[i, IDX[s[i + 2]]]) for i in range(4)],
            "probs": probs.round(4).tolist(),
        })
    return {"item_acc": item_ok / (4 * len(seqs)), "seq_acc": seq_ok / len(seqs), "rows": rows}


# ---------------------------------------------------------------------------
# Training (BP algorithm for multi-target outputs NNs, slide 27)
# ---------------------------------------------------------------------------

def train(hidden: int, lr: float, epochs: int, batch: int, l2: float, seed: int, log_every: int = 10):
    rng = np.random.default_rng(seed)
    p = init_params(hidden, rng)            # Step 2: initialise weights
    X, Y = encode(TRAIN)                    # Step 1: training data
    Xte, Yte = encode(TEST)
    hist = {"epoch": [], "train_loss": [], "test_loss": [], "train_seq_acc": [], "test_seq_acc": [],
            "train_item_acc": [], "test_item_acc": []}
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(TRAIN))
        for s in range(0, len(order), batch):   # Step 3: mini-batches
            idx = order[s:s + batch]
            H, Yhat = forward(p, X[idx])
            g = backward(p, X[idx], Y[idx], H, Yhat, l2)
            for k in p:                         # Step 4: update
                p[k] -= lr * g[k]
        if ep == 1 or ep % log_every == 0:
            tr, te = evaluate(p, TRAIN), evaluate(p, TEST)
            hist["epoch"].append(ep)
            hist["train_loss"].append(loss_fn(forward(p, X)[1], Y, p, 0.0))
            hist["test_loss"].append(loss_fn(forward(p, Xte)[1], Yte, p, 0.0))
            hist["train_seq_acc"].append(tr["seq_acc"])
            hist["test_seq_acc"].append(te["seq_acc"])
            hist["train_item_acc"].append(tr["item_acc"])
            hist["test_item_acc"].append(te["item_acc"])
    return p, hist                              # Step 5: repeat until convergence


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(p, hist, test_eval, seed_runs, fig_dir: Path, hidden: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, grey = "#2878B5", "#E07B39", "#667085"

    # 1. Unrolled architecture
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    example, xs = "AB1223", [1.3 + 2.0 * i for i in range(5)]
    for t, x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x - 0.45, 0.25), 0.9, 0.55, boxstyle="round,pad=0.03", fc="#EAF3F8", ec=blue))
        ax.text(x, 0.52, f"x{t+1} = {example[t]}", ha="center", va="center", fontsize=9)
        ax.add_patch(FancyBboxPatch((x - 0.55, 1.6), 1.1, 0.75, boxstyle="round,pad=0.03", fc="white", ec="#17324D", lw=1.5))
        ax.text(x, 1.97, f"h{t+1}\n({hidden} sigmoid)", ha="center", va="center", fontsize=8.5)
        tgt = t >= 1
        ax.add_patch(FancyBboxPatch((x - 0.45, 3.15), 0.9, 0.55, boxstyle="round,pad=0.03",
                                    fc="#E8F5EC" if tgt else "#F4F6F8", ec="#2E8B57" if tgt else grey,
                                    ls="-" if tgt else "--"))
        ax.text(x, 3.42, f"y{t+1} -> {example[t+1]}" if tgt else "y1: none", ha="center", va="center", fontsize=8.5)
        ax.add_patch(FancyArrowPatch((x, 0.82), (x, 1.58), arrowstyle="-|>", mutation_scale=12, color=blue))
        ax.add_patch(FancyArrowPatch((x, 2.37), (x, 3.13), arrowstyle="-|>", mutation_scale=12, color="#2E8B57"))
        ax.text(x + 0.08, 1.15, "W", fontsize=8, color=blue)
        ax.text(x + 0.08, 2.7, "V", fontsize=8, color="#2E8B57")
        if t < 4:
            ax.add_patch(FancyArrowPatch((x + 0.57, 1.97), (xs[t + 1] - 0.57, 1.97), arrowstyle="-|>", mutation_scale=12, color=orange))
            ax.text((x + xs[t + 1]) / 2, 2.07, "U", fontsize=8, color=orange, ha="center")
    ax.text(0.15, 1.97, "h0=0", fontsize=8, va="center", color=orange)
    ax.text(5.5, 4.15, "Unrolled multi-target outputs network (external input x_t, internal input h_{t-1}, one target per step)",
            ha="center", fontsize=10.5, weight="bold", color="#17324D")
    fig.savefig(fig_dir / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], color=blue, label="Training")
    axes[0].plot(hist["epoch"], hist["test_loss"], color=orange, label="Testing")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy cost (log scale)")
    axes[0].set_title("Cost over training")
    axes[0].legend(frameon=False)
    axes[1].plot(hist["epoch"], np.array(hist["train_seq_acc"]) * 100, color=blue, label="Training (sequence)")
    axes[1].plot(hist["epoch"], np.array(hist["test_seq_acc"]) * 100, color=orange, label="Testing (sequence)")
    axes[1].plot(hist["epoch"], np.array(hist["test_item_acc"]) * 100, color=orange, ls="--", alpha=0.7, label="Testing (item)")
    axes[1].set_ylim(-3, 103)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Free-running completion accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=200)
    plt.close(fig)

    # 3. Output probabilities for the test sequences
    rows = test_eval["rows"]
    fig, axes = plt.subplots(1, len(rows), figsize=(13, 3.4), sharey=True)
    for ax, r in zip(axes, rows):
        probs = np.array(r["probs"]).T          # [K,4]
        ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for j in range(4):
            for i in range(K):
                if probs[i, j] > 0.01:
                    ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if probs[i, j] > 0.6 else "black")
        ax.set_xticks(range(4), [f"item {j+3}" for j in range(4)], fontsize=7.5)
        ax.set_yticks(range(K), SYMBOLS)
        ax.set_title(f"{r['input']} -> {r['prediction'][2:]}  ({'OK' if r['correct'] else 'X'})", fontsize=9.5)
    fig.suptitle("Network output probabilities when completing each test sequence", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_output_probabilities.png", dpi=200)
    plt.close(fig)

    # 4. Hidden state after two items (PCA)
    allseqs = TRAIN + TEST
    h2 = np.array([complete(p, s[:2])[2][1] for s in allseqs])
    hc = h2 - h2.mean(0)
    _, _, vt = np.linalg.svd(hc, full_matrices=False)
    z = hc @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(z[:20, 0], z[:20, 1], s=70, color=blue, label="Training")
    ax.scatter(z[20:, 0], z[20:, 1], s=110, color=orange, marker="*", label="Testing")
    for (u, v), s in zip(z, allseqs):
        ax.annotate(s[:2], (u, v), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Internal state h2 after reading the first two items")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "hidden_state_pca.png", dpi=200)
    plt.close(fig)

    # 5. Robustness over random seeds
    accs = [r["test_seq_acc"] * 100 for r in seed_runs]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(accs)), accs, color=[blue if a == 100 else orange for a in accs])
    ax.set_xticks(range(len(accs)), [str(r["seed"]) for r in seed_runs], fontsize=8)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test sequence accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Test accuracy over {len(accs)} independent trainings (mean {np.mean(accs):.1f}%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "seed_robustness.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=20, help="extra runs for the robustness study")
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Gradient check (max relative error between BP and finite differences) ...")
    grad_err = gradient_check()
    print(f"  max relative error = {grad_err:.2e}")

    print(f"Training: hidden={args.hidden}, lr={args.lr}, epochs={args.epochs}, batch={args.batch}, seed={args.seed}")
    p, hist = train(args.hidden, args.lr, args.epochs, args.batch, args.l2, args.seed)
    tr, te = evaluate(p, TRAIN), evaluate(p, TEST)
    print(f"  training: sequence acc {tr['seq_acc']*100:.1f}%, item acc {tr['item_acc']*100:.1f}%")
    print(f"  testing : sequence acc {te['seq_acc']*100:.1f}%, item acc {te['item_acc']*100:.1f}%")
    print("\n  Input  Target   Predicted  Confidence (items 3-6)")
    for r in te["rows"]:
        conf = " ".join(f"{c:.3f}" for c in r["confidence"])
        print(f"  {r['input']}     {r['target']}   {r['prediction']}     {conf}   {'OK' if r['correct'] else 'WRONG'}")

    seed_runs = []
    for sd in range(args.n_seeds):
        q, _ = train(args.hidden, args.lr, args.epochs, args.batch, args.l2, sd, log_every=args.epochs)
        a, b = evaluate(q, TRAIN), evaluate(q, TEST)
        seed_runs.append({"seed": sd, "train_seq_acc": a["seq_acc"], "test_seq_acc": b["seq_acc"],
                          "test_item_acc": b["item_acc"]})
    mean_acc = np.mean([r["test_seq_acc"] for r in seed_runs])
    print(f"\nRobustness: mean test sequence accuracy over {args.n_seeds} seeds = {mean_acc*100:.1f}%")

    np.savez(out / "weights.npz", **p)
    make_figures(p, hist, te, seed_runs, out / "figures", args.hidden)
    results = {
        "config": vars(args), "symbols": SYMBOLS, "train": TRAIN, "test": TEST,
        "gradient_check_max_rel_error": grad_err,
        "train_metrics": {k: tr[k] for k in ("item_acc", "seq_acc")},
        "test_metrics": {k: te[k] for k in ("item_acc", "seq_acc")},
        "train_rows": tr["rows"], "test_rows": te["rows"],
        "final_train_loss": hist["train_loss"][-1], "final_test_loss": hist["test_loss"][-1],
        "history": hist, "seed_runs": seed_runs,
        "n_params": int(sum(v.size for v in p.values())),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved weights, results.json and figures to {out.resolve()}")


if __name__ == "__main__":
    main()
