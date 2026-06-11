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
    
    ann_data = [] # Collect data to space manually
    
    for seed in shared:
        ie = ema_seeds.index(seed); im = max_seeds.index(seed)
        ev = ema_accs[ie]; mv = max_accs[im]
        col = SEED_COLORS.get(seed, "#777")
        
        ax.plot([0, 1], [ev, mv], color=col, linewidth=1.5,
                marker="o", markersize=7, zorder=3, label=f"seed={seed}")
        
        # Save original Y position for spacing logic
        ann_data.append({
            "y_orig": mv, 
            "y_curr": mv, 
            "text": f"Δ={mv - ev:+.4f}", 
            "color": col
        })

    # 1. Sort annotations from bottom to top
    ann_data.sort(key=lambda d: d["y_orig"])

    # 2. Force a strict minimum vertical gap
    min_gap = 0.0018  # You can slightly increase this if needed
    for i in range(1, len(ann_data)):
        if ann_data[i]["y_curr"] - ann_data[i-1]["y_curr"] < min_gap:
            ann_data[i]["y_curr"] = ann_data[i-1]["y_curr"] + min_gap

    # 3. Plot the spaced text and draw connecting leader lines
    for d in ann_data:
        # Place text out at x=1.06 to clearly bypass the marker
        ax.text(1.06, d["y_curr"], d["text"], color=d["color"], 
                fontsize=7.5, va="center", ha="left")
        
        # Draw a faint leader line connecting the marker to the new text position
        ax.plot([1.015, 1.05], [d["y_orig"], d["y_curr"]], 
                color=d["color"], linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EMA\n(iFOPNG_ema)", "MAX\n(iFOPNG)"], fontsize=9)
    ax.set_ylabel("Avg. Accuracy (20 tasks)", fontsize=9)
    ax.set_title("Paired per-seed comparison", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.35, 1.7)
    
    # Expand Y limits slightly to account for text pushed upward
    ylo = min(min(ema_accs), min(max_accs)) - 0.003
    yhi = max(max(ema_accs), max(max_accs)) + 0.008
    if ann_data:
        yhi = max(yhi, ann_data[-1]["y_curr"] + 0.002)
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

    plt.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        plt.savefig(OUT + f"ema-vs-max-stat_14-15.{ext}")
    plt.close()
    
    print("Saved ema-vs-max-stat_14-15")
    print(f"\nEMA: {np.mean(ema_accs):.4f} +- {np.std(ema_accs):.4f}")
    print(f"MAX: {np.mean(max_accs):.4f} +- {np.std(max_accs):.4f}")
    print(f"Delta = {np.mean(diffs)*100:.3f} pp")
    print(f"Paired t(4) = {t_stat:.3f},  p = {p_t:.5f}")
    print(f"Wilcoxon W = {stat_w:.0f},  p = {p_w:.5f}")
    print(f"Cohen's d = {cohen_d:.3f}")

if __name__ == "__main__":
    main()