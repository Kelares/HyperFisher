"""
ema-vs-max-stat_14-15.py
─────────────────────────
Paired Wilcoxon + bar chart: EMA vs MAX Fisher accumulation.
20-task Permuted-MNIST, 5 seeds each. Sub-RQ4 statistical figure.

Output: plots/ema-vs-max-stat_14-15.png
"""
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from utils import STYLE

RESULTS = "results/"
OUT     = "plots/"
SEED_COLORS = {42: "#1B6CA8", 1234: "#E07B2A", 811: "#2E8B57",
               2137: "#8B5CF6", 111: "#C94040"}


def _load(fname):
    df = pd.read_csv(fname)
    accs, seeds = [], []
    for _, row in df.iterrows():
        try:
            s = ast.literal_eval(str(row["summary"]))
            c = ast.literal_eval(str(row["config"]))
        except Exception:
            continue
        acc = s.get("best/average_accuracy")
        tc  = s.get("task_completed", "?"); nt = c.get("num_tasks", "?")
        if acc and tc == nt:
            accs.append(acc); seeds.append(c.get("seed"))
    return accs, seeds


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    ema_accs, ema_seeds = _load(RESULTS + "414.csv")
    max_accs, max_seeds = _load(RESULTS + "415.csv")

    stat_w, p_w = stats.wilcoxon(max_accs, ema_accs, alternative="greater")
    t_stat, p_t = stats.ttest_rel(max_accs, ema_accs, alternative="greater")
    diffs   = [m - e for m, e in zip(max_accs, ema_accs)]
    cohen_d = np.mean(diffs) / np.std(diffs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Exp 14 vs 15 — Fisher Accumulation: EMA vs MAX  (20-task Permuted-MNIST, n=5 seeds)",
        fontsize=11, fontweight="bold",
    )

    # ── Left: paired lines ────────────────────────────────────────────────
    ax = axes[0]
    shared = sorted(set(ema_seeds) & set(max_seeds))
    for seed in shared:
        ie = ema_seeds.index(seed); im = max_seeds.index(seed)
        ev = ema_accs[ie]; mv = max_accs[im]
        col = SEED_COLORS.get(seed, "#777")
        ax.plot([0, 1], [ev, mv], color=col, linewidth=1.5,
                marker="o", markersize=7, zorder=3, label=f"seed={seed}")
        ax.annotate(f"  D={mv - ev:+.4f}", xy=(1, mv),
                    fontsize=7, color=col, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EMA\n(iFOPNG_ema)", "MAX\n(iFOPNG)"], fontsize=9)
    ax.set_ylabel("Avg. Accuracy (20 tasks)", fontsize=9)
    ax.set_title("Paired per-seed comparison", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.35, 1.7)
    ylo = min(min(ema_accs), min(max_accs)) - 0.003
    yhi = max(max(ema_accs), max(max_accs)) + 0.008
    ax.set_ylim(ylo, yhi)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    # ── Right: bar + significance bracket ─────────────────────────────────
    ax2 = axes[1]
    means = [np.mean(ema_accs), np.mean(max_accs)]
    stds  = [np.std(ema_accs),  np.std(max_accs)]
    ax2.bar([0, 1], means, yerr=stds,
            color=["#A8C7E8", "#1B6CA8"], width=0.5, capsize=5, zorder=3,
            alpha=0.88,
            error_kw={"linewidth": 1.2, "ecolor": "#333", "capthick": 1.2})
    for lst, xi in [(ema_accs, 0), (max_accs, 1)]:
        jit = np.linspace(-0.09, 0.09, len(lst))
        for j, v in zip(jit, lst):
            ax2.scatter(xi + j, v, color="white", s=20, zorder=4,
                        edgecolors="#333", linewidths=0.8)
    y_top = max(means) + max(stds) + 0.002
    ax2.plot([0, 0, 1, 1], [y_top, y_top+0.001, y_top+0.001, y_top],
             color="#333", linewidth=1.0)
    star = ("***" if p_t < 0.001 else "**" if p_t < 0.01
            else "*" if p_t < 0.05 else "ns")
    ax2.text(0.5, y_top+0.0015, star, ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["EMA\n(iFOPNG_ema)", "MAX\n(iFOPNG)"], fontsize=9)
    ax2.set_ylabel("Avg. Accuracy (20 tasks)", fontsize=9)
    ax2.set_title("Mean +- std with significance", fontsize=10, fontweight="bold")
    ax2.set_ylim(ylo - 0.001, y_top + 0.006)
    stats_txt = (
        f"Paired t(4):  t = {t_stat:.2f},  p = {p_t:.5f}\n"
        f"Wilcoxon:     W = {stat_w:.0f},    p = {p_w:.5f}\n"
        f"Cohen's d (paired) = {cohen_d:.2f}\n"
        f"Mean D = {np.mean(diffs)*100:.2f} pp  (MAX > EMA)\n"
        f"H0: EMA >= MAX  ->  REJECTED"
    )
    ax2.text(0.05, 0.5, stats_txt, transform=ax2.transAxes,
             fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#ccc", alpha=0.95))

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "ema-vs-max-stat_14-15.png")
    plt.close()
    print("Saved ema-vs-max-stat_14-15.png")
    print(f"\nEMA: {np.mean(ema_accs):.4f} +- {np.std(ema_accs):.4f}")
    print(f"MAX: {np.mean(max_accs):.4f} +- {np.std(max_accs):.4f}")
    print(f"Delta = {np.mean(diffs)*100:.3f} pp")
    print(f"Paired t(4) = {t_stat:.3f},  p = {p_t:.5f}")
    print(f"Wilcoxon W = {stat_w:.0f},  p = {p_w:.5f}")
    print(f"Cohen's d = {cohen_d:.3f}")

if __name__ == "__main__":
    main()