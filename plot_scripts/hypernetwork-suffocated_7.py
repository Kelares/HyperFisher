"""
hypernetwork-suffocated_7.py
─────────────────────────────
Bar chart for Exp 7 — Split-MNIST SH HN suffocated (dh=8, 1406 chunks).
All methods × 5-7 seeds. Sub-RQ1 core result.

Output: plots/hypernetwork-suffocated_7.png
"""
import os
import matplotlib.pyplot as plt
from utils import STYLE, METHOD_ORDER, load_exp, bar_panel, COLORS

RESULTS = "results/"
OUT     = "plots/"

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    data = load_exp(RESULTS + "407.csv", min_seeds=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Exp 7 — Split-MNIST HyperNetwork Suffocated  (dh=8, 5T×15ep, AdamW first-task)",
        fontsize=11, fontweight="bold",
    )

    bar_panel(axes[0], data, "acc", "Avg. Accuracy", "Average Accuracy",
              ylim=(0.45, 1.02))
    bar_panel(axes[1], data, "bwt", "BWT", "Backward Transfer",
              ylim=(-0.45, 0.05))

    # Mark Adam line on accuracy panel
    if "adam" in data:
        adam_mean = data["adam"]["acc_mean"]
        axes[0].axhline(adam_mean, color=COLORS["adam"], lw=1.0, ls=":", alpha=0.6)
        axes[0].text(0.99, adam_mean + 0.005, f"Adam {adam_mean:.3f}",
                     ha="right", va="bottom",
                     transform=axes[0].get_yaxis_transform(),
                     fontsize=7.5, color=COLORS["adam"])

    axes[0].text(0.02, 0.04, "// = fewer than 3 seeds",
                 transform=axes[0].transAxes, fontsize=7, color="#666", style="italic")

    plt.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        plt.savefig(OUT + f"hypernetwork-suffocated_7.{ext}")
    plt.close()
    print("Saved hypernetwork-suffocated_7")

if __name__ == "__main__":
    main()