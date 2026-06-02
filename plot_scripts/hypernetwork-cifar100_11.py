"""
hypernetwork-cifar100_11.py
────────────────────────────
Bar chart for Exp 11 — Split-CIFAR100 HN standard.
All methods × 5 seeds. OGD uses reduced lr=10⁻³ (no Fisher preconditioning).

Output: plots/hypernetwork-cifar100_11.png
"""
import os
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, bar_panel

RESULTS = "results/"
OUT     = "plots/"

def skip_ogd_high_lr(method, config, summary):
    """Keep only OGD runs with lr <= 0.001 (lr=0.05 explodes without Fisher)."""
    if method == "ogd":
        return config.get("lr", 999) > 0.001
    return False

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    data = load_exp(RESULTS + "411.csv", min_seeds=3, skip_fn=skip_ogd_high_lr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Exp 11 — Split-CIFAR100 HN Standard  (10T×50ep, AdamW first-task @ 10⁻³)",
        fontsize=11, fontweight="bold",
    )

    bar_panel(axes[0], data, "acc", "Avg. Accuracy", "Average Accuracy",
              ylim=(0.05, 0.60))
    bar_panel(axes[1], data, "bwt", "BWT", "Backward Transfer",
              ylim=(-0.20, 0.05))

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "hypernetwork-cifar100_11.png")
    plt.close()
    print("Saved hypernetwork-cifar100_11.png")

if __name__ == "__main__":
    main()