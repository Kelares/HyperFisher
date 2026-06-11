"""
standalone-cifar100_6.py
─────────────────────────
Bar chart for Exp 6 — Split-CIFAR100 MH Standalone.
All methods × 3 seeds, SGD first-task @ 10⁻². EWC lam=10 (lam=50 crashed).

Output: plots/standalone-cifar100_6.png
"""
import os
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, bar_panel, skip_exp6, COLORS

RESULTS = "results/"
OUT     = "plots/"

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    data = load_exp(RESULTS + "406.csv", skip_fn=skip_exp6)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Exp 6 — Split-CIFAR100 MH Standalone  (10T×10ep, SGD first-task @ 10⁻²)",
                 fontsize=11, fontweight="bold")

    bar_panel(axes[0], data, "acc", "Avg. Accuracy", "Average Accuracy",
              ylim=(0.10, 0.62))
    bar_panel(axes[1], data, "bwt", "BWT", "Backward Transfer",
              ylim=(-0.55, 0.05))

    # Annotate EWC lambda deviation
    if "ewc" in data:
        from utils import METHOD_ORDER
        sorted_methods = sorted(data, key=lambda m: -(data[m]["acc_mean"] or 0))
        xi = sorted_methods.index("ewc")
        axes[0].annotate(
            "lam=10\n(lam=50 crashed)",
            xy=(xi, data["ewc"]["acc_mean"]),
            xytext=(xi + 0.6, data["ewc"]["acc_mean"] + 0.07),
            fontsize=6.5, color=COLORS["ewc"],
            arrowprops=dict(arrowstyle="->", color=COLORS["ewc"], lw=0.8),
        )

    plt.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        plt.savefig(OUT + f"standalone-cifar100_6.{ext}")
    plt.close()
    print("Saved standalone-cifar100_6")

if __name__ == "__main__":
    main()