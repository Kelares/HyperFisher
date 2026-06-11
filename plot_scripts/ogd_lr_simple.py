"""
ogd_lr_sweep_simple.py
──────────────────────
Simple 2-panel bar chart for the OGD learning rate sweep on
Split-CIFAR100 HN (Exp 11). Shows average accuracy and BWT
for the two viable rates; marks lr=0.005 as failed.

Output: plots/ogd-lr-sweep-simple_11.pdf
         plots/ogd-lr-sweep-simple_11.png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb

ENTITY   = "michalowski-jb-tilburg-university"
PROJECT  = "HyperFisher"
EXP_ID   = 411       # Split-CIFAR100 HN
OUT_DIR  = "plots/"

matplotlib.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

LR_ORDER   = [5e-4, 1e-3, 5e-3]
LR_LABELS  = [r"$5\times10^{-4}$", r"$10^{-3}$\;(selected)", r"$5\times10^{-3}$"]
LR_COLORS  = ["#4878CF", "#2ca02c", "#d62728"]
FAILED_LR  = 5e-3

# ─────────────────────────────────────────────────────────────────────────────

def fetch():
    api  = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={
            "config.experiment_id": EXP_ID,
            "config.methods":       ["ogd"],
        },
    )
    data = {lr: {"acc": [], "bwt": []} for lr in LR_ORDER}
    seen = set()
    for run in runs:
        if run.state != "finished":
            continue
        lr   = run.config.get("lr")
        seed = run.config.get("seed")
        if (lr, seed) in seen:
            continue
        seen.add((lr, seed))
        s    = dict(run.summary)
        acc  = s.get("best/average_accuracy")
        bwt  = s.get("best/bwt")
        if acc is None or lr not in data:
            continue
        data[lr]["acc"].append(acc)
        data[lr]["bwt"].append(bwt if bwt is not None else float("nan"))
    return data


def plot(data):
    fig, (ax_acc, ax_bwt) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    fig.suptitle(
        r"\texttt{OGD} learning rate sweep --- Split-CIFAR100 HN (Exp~11)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    x = np.arange(len(LR_ORDER))
    width = 0.55

    for ax, metric, ylabel, title in [
        (ax_acc, "acc", "Average Accuracy", "Average Accuracy"),
        (ax_bwt, "bwt", "BWT",              "Backward Transfer"),
    ]:
        ax.set_title(title, pad=5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(LR_LABELS, fontsize=9)
        ax.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "bwt":
            ax.axhline(0.0, color="#888", lw=0.8, ls="--", zorder=0)

        for xi, (lr, col) in enumerate(zip(LR_ORDER, LR_COLORS)):
            vals = data[lr][metric]

            if lr == FAILED_LR or len(vals) == 0:
                # draw a hatched "failed" bar
                height = ax_acc.get_ylim()[1] * 0.15 if metric == "acc" else -0.02
                bar = ax.bar(xi, abs(height) if metric == "acc" else -height,
                             width=width, color="#ffcccc",
                             edgecolor="#d62728", linewidth=1.2,
                             hatch="///", zorder=2)
                ax.text(xi, 0.005 if metric == "acc" else -0.005,
                        "all seeds\nfailed",
                        ha="center", va="bottom" if metric == "acc" else "top",
                        fontsize=8, color="#d62728", zorder=3)
                continue

            vals  = np.array(vals)
            mean  = vals.mean()
            std   = vals.std()
            ax.bar(xi, mean, width=width, color=col,
                   alpha=0.85, zorder=2)
            ax.errorbar(xi, mean, yerr=std,
                        fmt="none", color="#333", capsize=4,
                        linewidth=1.0, capthick=1.0, zorder=3)
            # seed scatter
            jit = np.linspace(-0.08, 0.08, len(vals))
            for j, v in zip(jit, vals):
                ax.scatter(xi + j, v, color="white", s=18,
                           edgecolors="#333", linewidths=0.6, zorder=4)
            ax.text(xi, mean + std + 0.005 * (1 if metric == "acc" else -1),
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom" if metric == "acc" else "top",
                    fontsize=8.5, color="#222", zorder=4)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{OUT_DIR}ogd-lr-sweep-simple_11.{ext}"
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Fetching OGD LR sweep data...")
    data = fetch()
    for lr, d in data.items():
        print(f"  lr={lr}: n={len(d['acc'])}  "
              f"acc={np.mean(d['acc'])*100:.1f}%" if d['acc'] else f"  lr={lr}: FAILED")
    plot(data)
    print("Done.")