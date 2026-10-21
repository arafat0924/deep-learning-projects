"""Hyperparameter selection by leave-one-out cross-validation on the TRAINING set only.

Each training sequence is held out in turn, the network is trained on the other
19, and the held-out sequence is completed from its first two items. The test
set is never used here, so the chosen configuration is not tuned on test data.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import seq_completion as sc


def loo_accuracy(hidden: int, l2: float, lr: float, epochs: int, batch: int, seed: int) -> float:
    full = list(sc.TRAIN)
    correct = 0
    try:
        for i, held in enumerate(full):
            sc.TRAIN[:] = full[:i] + full[i + 1:]
            p, _ = sc.train(hidden, lr, epochs, batch, l2, seed, log_every=10 ** 9)
            correct += sc.complete(p, held[:2])[0] == held
    finally:
        sc.TRAIN[:] = full
    return correct / len(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3])
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/model_selection.json")
    args = ap.parse_args()

    rows = []
    for h, l2 in itertools.product(args.hidden, args.l2):
        acc = loo_accuracy(h, l2, args.lr, args.epochs, args.batch, args.seed)
        rows.append({"hidden": h, "l2": l2, "loo_seq_acc": acc})
        print(f"hidden={h:3d}  l2={l2:.0e}  LOO sequence accuracy = {acc*100:5.1f}%", flush=True)

    # Best LOO accuracy; ties broken by the smaller network, then the weaker penalty.
    best = max(rows, key=lambda r: (r["loo_seq_acc"], -r["hidden"], -r["l2"]))
    print(f"\nSelected: hidden={best['hidden']}, l2={best['l2']:.0e}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "grid": rows, "selected": best}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
