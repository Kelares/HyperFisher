"""
standalone-mnist_1-2.py
────────────────────────
Bar charts for Exp 1 (Permuted-MNIST) and Exp 2 (Split-MNIST MH).
All methods × 3-5 seeds. Sub-RQ3 Panel A.

Output: plots/standalone-mnist_1-2.png
"""
import os
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, bar_panel

RESULTS = "results/"
OUT     = "plots/"

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    data1 = load_exp(RESULTS + "401.csv")
    data2 = load_exp(RESULTS + "402.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Standalone Target-Network Benchmarks — MNIST",
                 fontsize=12, fontweight="bold", y=1.01)

    bar_panel(axes[0,0], data1, "acc", "Avg. Accuracy",
              "Exp 1 — Permuted-MNIST (5T)", ylim=(0.45, 0.95))
    bar_panel(axes[0,1], data1, "bwt", "BWT",
              "Exp 1 — Permuted-MNIST (5T) BWT", ylim=(-0.55, 0.05))
    bar_panel(axes[1,0], data2, "acc", "Avg. Accuracy",
              "Exp 2 — Split-MNIST MH (5T)", ylim=(0.75, 1.02))
    bar_panel(axes[1,1], data2, "bwt", "BWT",
              "Exp 2 — Split-MNIST MH (5T) BWT", ylim=(-0.35, 0.05))

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "standalone-mnist_1-2.png")
    plt.close()
    print("Saved standalone-mnist_1-2.png")

if __name__ == "__main__":
    main()