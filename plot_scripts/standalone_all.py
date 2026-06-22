"""
standalone_all_appendix.py
──────────────────────────
One combined figure: bar charts (Avg. Accuracy + BWT) for all five
standalone target-network benchmarks.

Benchmarks / CSV source:
  Row 0  Permuted-MNIST       401.csv
  Row 1  Split-MNIST MH       402.csv   (seed 2136 excluded — typo run)
  Row 2  Split-MNIST SH       403.csv
  Row 3  Split-CIFAR10        404.csv
  Row 4  Split-CIFAR100       406.csv

Layout: 5 rows × 2 columns  (col 0 = Acc, col 1 = BWT)
Output: plots/standalone_all_appendix.{pdf,png}
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/"
OUT_DIR     = "plots/"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style (matches existing project scripts) ──────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "figure.dpi":          150,
    "savefig.dpi":         200,
    "savefig.bbox":        "tight",
    "axes.grid":           True,
    "grid.alpha":          0.2,
    "grid.linewidth":      0.5,
    "text.usetex":         False,
})

METHOD_COLORS = {
    "ifopng": "#1B6CA8",
    "fopng":  "#5BA3D9",
    "ewc":    "#2E8B57",
    "ogd":    "#E07B2A",
    "ong":    "#C94040",
    "fng":    "#8B5CF6",
    "sgd":    "#888888",
    "adam":   "#444444",
}
METHOD_LABELS = {
    "ifopng": "iFOPNG",
    "fopng":  "FOPNG",
    "ewc":    "EWC",
    "ogd":    "OGD",
    "ong":    "ONG",
    "fng":    "FNG",
    "sgd":    "SGD",
    "adam":   "Adam",
}
# Consistent left-to-right ordering across every panel
ORDER = ["ifopng", "fopng", "ogd", "ong", "fng", "ewc", "adam", "sgd"]


# ── Data loading ──────────────────────────────────────────────────────────────
def safe_parse(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return {}


def load_exp(fname, min_acc=0.05, min_seeds=2, excl_seeds=None):
    """
    Parse a W&B CSV export.

    Returns dict[method] -> {
        "accs": list[float], "bwts": list[float],
        "acc_mean", "acc_std", "bwt_mean", "bwt_std": float
    }

    Per-seed best run is selected (handles multiple runs of same seed).
    Rows whose seed is in `excl_seeds` are silently dropped.
    """
    excl_seeds = set(excl_seeds or [])
    df = pd.read_csv(fname)
    # seed_best[method][seed] = (best_acc, bwt)
    seed_best = defaultdict(lambda: defaultdict(lambda: (-999.0, None)))

    for _, row in df.iterrows():
        s = safe_parse(row["summary"])
        c = safe_parse(row["config"])

        method = c.get("methods", ["?"])
        if isinstance(method, list):
            method = method[0]

        seed = c.get("seed", "?")
        if seed in excl_seeds:
            continue

        acc = s.get("best/average_accuracy", None)
        bwt = s.get("best/bwt", None)
        tc  = s.get("task_completed", "?")
        nt  = c.get("num_tasks", "?")

        # Only accept completed runs above the noise floor
        if acc is None or tc != nt or acc <= min_acc:
            continue

        if acc > seed_best[method][seed][0]:
            seed_best[method][seed] = (acc, bwt)

    result = {}
    for method, seeds in seed_best.items():
        accs = [v[0] for v in seeds.values()]
        bwts = [v[1] for v in seeds.values() if v[1] is not None]
        if len(accs) < min_seeds:
            continue
        result[method] = {
            "accs":     accs,
            "bwts":     bwts,
            "acc_mean": float(np.mean(accs)),
            "acc_std":  float(np.std(accs)),
            "bwt_mean": float(np.mean(bwts)) if bwts else None,
            "bwt_std":  float(np.std(bwts))  if bwts else None,
        }
    return result


# ── Plotting primitive ────────────────────────────────────────────────────────
def bar_panel(ax, data, metric, ylabel, ylim, title):
    """
    Draw a single bar panel.

    `metric` must be "acc" or "bwt".
    Methods are sorted by descending mean within the panel.
    A hatch is added when n < 3 seeds (flags incomplete coverage).
    """
    methods = [
        m for m in ORDER
        if m in data and data[m][f"{metric}_mean"] is not None
    ]
    # Sort by descending mean so the best method sits leftmost
    methods = sorted(methods, key=lambda m: -(data[m][f"{metric}_mean"] or 0))

    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        d    = data[m]
        mean = d[f"{metric}_mean"]
        std  = d[f"{metric}_std"] or 0.0
        vals = d[f"{metric}s"]
        col  = METHOD_COLORS.get(m, "#999999")
        n    = len(vals)

        ax.bar(
            i, mean,
            yerr=std,
            color=col,
            width=0.6,
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "#333333", "capthick": 1.0},
            zorder=3,
            alpha=0.88,
            hatch="/" if n < 3 else "",
        )

        # Scatter individual seed values
        jitter = np.linspace(-0.12, 0.12, n)
        for j, v in zip(jitter, vals):
            ax.scatter(
                i + j, v,
                color="white", s=12, zorder=4,
                edgecolors="#333333", linewidths=0.6,
            )

    ax.axhline(0, color="#333333", lw=0.7, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    if ylim:
        ax.set_ylim(ylim)


# ── Experiment registry ───────────────────────────────────────────────────────
# (label, csv_filename, excl_seeds, acc_ylim, bwt_ylim)
EXPERIMENTS = [
    (
        "Permuted-MNIST  (5T)",
        "401.csv",
        None,
        (0.50, 0.97),
        (-0.52, 0.07),
    ),
    (
        "Split-MNIST MH  (5T)",
        "402.csv",
        [2136],          # typo run — canonical seed set is {42,111,811,1234,2137}
        (0.75, 1.03),
        (-0.22, 0.05),
    ),
    (
        "Split-MNIST SH  (5T)",
        "403.csv",
        None,
        (0.45, 0.92),
        (-0.57, 0.07),
    ),
    (
        "Split-CIFAR10   (5T)",
        "404.csv",
        None,
        (0.55, 0.92),
        (-0.40, 0.07),
    ),
    (
        "Split-CIFAR100  (10T)",
        "406.csv",
        None,
        (0.10, 0.62),
        (-0.48, 0.07),
    ),
]

# ── Build figure ──────────────────────────────────────────────────────────────
N_ROWS = len(EXPERIMENTS)
fig, axes = plt.subplots(
    N_ROWS, 2,
    figsize=(11, 3.6 * N_ROWS),
)
fig.suptitle(
    "Standalone Target-Network — All Benchmarks\n"
    "Bar height: mean over seeds; error bars: ±1 s.d.; dots: individual seeds",
    fontsize=11, fontweight="bold", y=1.005,
)

for row_i, (label, fname, excl, acc_ylim, bwt_ylim) in enumerate(EXPERIMENTS):
    data = load_exp(
        RESULTS_DIR + fname,
        excl_seeds=excl,
    )
    if not data:
        print(f"  WARNING: no data loaded for {fname}")
        continue

    # Seed-count note (for caption / sanity check)
    n_seeds = {m: len(v["accs"]) for m, v in data.items()}
    print(f"{label}: {n_seeds}")

    bar_panel(
        axes[row_i, 0], data, "acc",
        "Avg. Accuracy",
        acc_ylim,
        f"{label}  —  Accuracy",
    )
    bar_panel(
        axes[row_i, 1], data, "bwt",
        "BWT",
        bwt_ylim,
        f"{label}  —  Backward Transfer",
    )

# ── Single shared legend (bottom centre) ─────────────────────────────────────
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m], alpha=0.88)
    for m in ORDER
    if m in METHOD_COLORS
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(ORDER),
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.015),
)

plt.tight_layout(rect=[0, 0.03, 1, 1], pad=1.8, h_pad=2.5)

for ext in ("pdf", "png"):
    out_path = OUT_DIR + f"standalone_all_appendix.{ext}"
    plt.savefig(out_path)
    print(f"Saved {out_path}")

plt.close()