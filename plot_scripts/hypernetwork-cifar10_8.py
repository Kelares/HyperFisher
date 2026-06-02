"""
hypernetwork-cifar10_8.py
──────────────────────────
Bar chart for Exp 8 — Split-CIFAR10 HN standard.
All methods × 5 seeds, AdamW first-task @ 10⁻³. Sub-RQ1 primary result.

Output: plots/hypernetwork-cifar10_8.png
"""
import os
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, bar_panel

RESULTS = "results/"
OUT     = "plots/"

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    data = load_exp(RESULTS + "408.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Exp 8 — Split-CIFAR10 HN Standard  (5T×50ep, AdamW first-task @ 10⁻³)",
        fontsize=11, fontweight="bold",
    )

    bar_panel(axes[0], data, "acc", "Avg. Accuracy", "Average Accuracy",
              ylim=(0.55, 1.0))
    bar_panel(axes[1], data, "bwt", "BWT", "Backward Transfer",
              ylim=(-0.12, 0.02))

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "hypernetwork-cifar10_8.png")
    plt.close()
    print("Saved hypernetwork-cifar10_8.png")

if __name__ == "__main__":
    main()