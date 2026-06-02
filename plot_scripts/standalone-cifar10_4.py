"""
standalone-cifar10_4.py
───────────────────────
Bar chart for Exp 4 — Split-CIFAR10 MH Standalone.
All methods × 5 seeds, Adam first-task @ 10⁻³. Sub-RQ3 Panel A.

Output: plots/standalone-cifar10_4.png
"""
import os
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, bar_panel

RESULTS = "results/"
OUT     = "plots/"

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    data = load_exp(RESULTS + "404.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Exp 4 — Split-CIFAR10 MH Standalone  (5T×5ep, Adam first-task @ 10⁻³)",
                 fontsize=11, fontweight="bold")

    bar_panel(axes[0], data, "acc", "Avg. Accuracy", "Average Accuracy",
              ylim=(0.55, 0.95))
    bar_panel(axes[1], data, "bwt", "BWT", "Backward Transfer",
              ylim=(-0.40, 0.05))

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "standalone-cifar10_4.png")
    plt.close()
    print("Saved standalone-cifar10_4.png")

if __name__ == "__main__":
    main()