"""
plot_trajectories.py
────────────────────
General-purpose trajectory plotter for any HyperFisher experiment.
Reads exclusively from run.summary (no scan_history).

USAGE
  Plot a specific experiment:
      python plot_trajectories.py --exp 402
      python plot_trajectories.py --exp 402 411
      python plot_trajectories.py --exp all          # every registered config

ADD A NEW EXPERIMENT
  Add one entry to EXPERIMENT_REGISTRY below — that's it.
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as stats
import wandb

# ══════════════════════════════════════════════════════════════════════════════
# WANDB CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════════
ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"
OUT_DIR = "visualizations/trajectories/"
EXT     = ["pdf", "png"]

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT REGISTRY
# Add an entry here whenever you run a new config.
# y_lim is (min, max) for the accuracy axis — tune per benchmark.
# y_ticks is the list of gridlines/tick marks shown.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExpConfig:
    exp_id:   int
    label:    str            # figure title / suptitle
    n_tasks:  int
    seeds:    List[int]
    out_stem: str            # filename without extension
    y_lim:    tuple  = (0.60, 0.97)
    y_ticks:  list   = field(default_factory=lambda: [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


EXPERIMENT_REGISTRY = {
    402: ExpConfig(
        exp_id=402,
        label="C2 · Split-MNIST MH · Standalone",
        n_tasks=5,
        seeds=[42, 1234, 811],
        out_stem="c2_split_mnist_mh_standalone",
        y_lim=(0.70, 1.00),
        y_ticks=[0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    ),
    404: ExpConfig(
        exp_id=404,
        label="C4 · Split-CIFAR10 MH · Standalone",
        n_tasks=5,
        seeds=[42, 1234, 811],
        out_stem="c4_split_cifar10_mh_standalone",
        y_lim=(0.50, 0.97),
        y_ticks=[0.55, 0.65, 0.75, 0.85, 0.95],
    ),
    406: ExpConfig(
        exp_id=406,
        label="C6 · Split-CIFAR100 MH · Standalone",
        n_tasks=10,
        seeds=[42, 1234, 811],
        out_stem="c6_split_cifar100_mh_standalone",
        y_lim=(0.20, 0.80),
        y_ticks=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    ),
    408: ExpConfig(
        exp_id=408,
        label="C8 · Split-CIFAR10 · Standard HN",
        n_tasks=5,
        seeds=[42, 1234, 811],
        out_stem="c8_split_cifar10_standard_hn",
        y_lim=(0.60, 0.97),
        y_ticks=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    ),
    411: ExpConfig(
        exp_id=411,
        label="C11 · Split-CIFAR100 · Standard HN",
        n_tasks=10,
        seeds=[42, 1234, 811],
        out_stem="c11_split_cifar100_standard_hn",
        y_lim=(0.20, 0.80),
        y_ticks=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# TYPESETTING  (NeurIPS / ICML single-column)
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "legend.frameon":     True,
    "legend.fancybox":    False,
    "legend.edgecolor":   "#bbbbbb",
    "figure.dpi":         300,
})

# ══════════════════════════════════════════════════════════════════════════════
# METHOD STYLES
# Projection-based  → solid,   mid weight
# Regularisation    → dashed,  subordinate
# Naive baselines   → dotted,  subordinate
# iFOPNG (ours)     → thick solid, always on top, distinct red
# ══════════════════════════════════════════════════════════════════════════════
METHOD_STYLES = {
    "EFOPNG":   {"color": "#B31B1B", "linestyle": "-",  "linewidth": 2.2,
                 "fill_alpha": 0.18, "label": "iFOPNG (Ours)"},
    "FOPNG":    {"color": "#1F77B4", "linestyle": "-",  "linewidth": 1.4,
                 "fill_alpha": 0.07, "label": "FOPNG"},
    "PREFOPNG": {"color": "#FF7F0E", "linestyle": "-",  "linewidth": 1.4,
                 "fill_alpha": 0.07, "label": "FOPNG-PreFisher"},
    "OGD":      {"color": "#5A5A5A", "linestyle": "-",  "linewidth": 1.4,
                 "fill_alpha": 0.07, "label": "OGD"},
    "ONG":      {"color": "#BCBD22", "linestyle": "-",  "linewidth": 1.4,
                 "fill_alpha": 0.07, "label": "ONG"},
    "FNG":      {"color": "#2CA02C", "linestyle": "--", "linewidth": 1.2,
                 "fill_alpha": 0.05, "label": "FNG"},
    "EWC":      {"color": "#9467BD", "linestyle": "--", "linewidth": 1.2,
                 "fill_alpha": 0.05, "label": "EWC"},
    "ADAM":     {"color": "#D62728", "linestyle": ":",  "linewidth": 1.2,
                 "fill_alpha": 0.05, "label": "Adam"},
    "SGD":      {"color": "#E377C2", "linestyle": ":",  "linewidth": 1.2,
                 "fill_alpha": 0.05, "label": "SGD"},
}

# Baselines rendered first so iFOPNG always lands on top
DRAW_ORDER = [m for m in METHOD_STYLES if m != "EFOPNG"] + ["EFOPNG"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING  (summary-only, no scan_history)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_experiment(cfg: ExpConfig) -> pd.DataFrame:
    """
    Pull all runs for `cfg.exp_id` from wandb and return a tidy DataFrame:
        Method | Seed | t | avg_acc
    where avg_acc = mean accuracy over the first t tasks at checkpoint t.
    """
    api  = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.experiment_id": cfg.exp_id},
    )
    print(f"\n[C{cfg.exp_id}]  {len(runs)} runs found")

    # ------------------------------------------------------------------
    # Extract per-method, per-seed trajectory from run.summary
    # Key tried in order:
    #   1. "{config_value}/results"  — written by main.py with the CLI arg
    #   2. "best/results"            — fallback written by main.py
    # ------------------------------------------------------------------
    best: dict = {}   # best[method][seed][t] = best avg_acc seen across duplicate runs

    for run in runs:
        rcfg   = run.config
        method = rcfg.get("methods", [None])
        if isinstance(method, list):
            method = method[0] if method else None
        if method is None:
            continue

        cfg_val = str(method).lower()
        display = cfg_val.upper()           # EFOPNG / ADAM / SGD …
        seed    = rcfg.get("seed")
        if seed not in cfg.seeds:
            continue

        # --- locate results dict ---
        s       = run.summary
        results = s.get(f"{cfg_val}/results") or s.get("best/results")
        if not results or "acc" not in results:
            print(f"  [warn] {display} seed={seed}: no results in summary (run {run.id})")
            continue

        raw_acc = results["acc"]   # {"1": [...], "2": [...], ...}

        # --- accumulate, keep best per (method, seed, t) ---
        best.setdefault(display, {}).setdefault(seed, {})
        for tc_str, acc_list in raw_acc.items():
            tc      = int(tc_str)
            avg_acc = float(np.mean(acc_list[:tc]))
            if avg_acc > 1.0:
                avg_acc /= 100.0
            prev = best[display][seed].get(tc, -np.inf)
            if avg_acc >= prev:
                best[display][seed][tc] = avg_acc

    # ------------------------------------------------------------------
    # Flatten to DataFrame rows
    # ------------------------------------------------------------------
    rows = []
    for display, seed_dict in best.items():
        for seed, t_dict in seed_dict.items():
            for t, avg_acc in t_dict.items():
                rows.append({"Method": display, "Seed": seed,
                             "t": t, "avg_acc": avg_acc})

    if not rows:
        print(f"  [ERROR] No data loaded for experiment {cfg.exp_id}.")
        return pd.DataFrame(columns=["Method", "Seed", "t", "avg_acc"])

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Seed-count validation
    # ------------------------------------------------------------------
    for method, mdf in df.groupby("Method"):
        n_seeds = mdf["Seed"].nunique()
        if n_seeds < len(cfg.seeds):
            missing = set(cfg.seeds) - set(mdf["Seed"].unique())
            print(f"  ⚠  {method}: {n_seeds}/{len(cfg.seeds)} seeds "
                  f"(missing {sorted(missing)})")
        else:
            print(f"  ✓  {method}: {n_seeds} seeds")

    print(f"  → {len(df)} trajectory points total")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_panel(ax: plt.Axes, df: pd.DataFrame,
               cfg: ExpConfig, show_ylabel: bool = True) -> list:
    """
    Draw one trajectory panel (avg accuracy vs tasks) onto `ax`.
    Uses 68 % CI (≈ ±1 SE) from a t-distribution — appropriate for 3–5 seeds.
    Returns legend handle list.
    """
    ax.tick_params(direction="in", top=True, right=True, which="both")

    handles = []
    for method_name in DRAW_ORDER:
        sub = df[df["Method"] == method_name]
        if sub.empty:
            continue

        style = METHOD_STYLES.get(
            method_name,
            {"color": "#555555", "linestyle": "-", "linewidth": 1.2,
             "fill_alpha": 0.06, "label": method_name.capitalize()},
        )

        grouped    = sub.groupby("t")["avg_acc"]
        mean_vals  = grouped.mean()
        sem_vals   = grouped.sem().fillna(0)
        n_vals     = grouped.count()
        xs         = mean_vals.index.values

        # 68 % CI from t-distribution  (ppf(0.84) ≈ ppf(1 - 0.16))
        t_crit = stats.t.ppf(0.84, df=np.maximum(n_vals - 1, 1))
        margin = t_crit * sem_vals
        margin[n_vals <= 1] = 0.0

        lo = np.clip(mean_vals.values - margin.values, *cfg.y_lim)
        hi = np.clip(mean_vals.values + margin.values, *cfg.y_lim)

        ax.fill_between(xs, lo, hi,
                        color=style["color"],
                        alpha=style["fill_alpha"],
                        linewidth=0)

        ax.plot(xs, mean_vals.values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=3 if method_name == "EFOPNG" else 2)

        handles.append(mlines.Line2D(
            [], [],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        ))

    ax.set_xlabel("Number of Tasks Trained", labelpad=3)
    if show_ylabel:
        ax.set_ylabel("Avg Accuracy on Trained Tasks", labelpad=3)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.set_xticks(range(1, cfg.n_tasks + 1))
    ax.set_xlim(0.8, cfg.n_tasks + 0.2)
    ax.set_ylim(*cfg.y_lim)
    ax.set_yticks(cfg.y_ticks)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("black")

    return handles


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: build one figure per experiment
# ══════════════════════════════════════════════════════════════════════════════
def plot_experiment(cfg: ExpConfig) -> None:
    df = fetch_experiment(cfg)
    if df.empty:
        print(f"  Skipping figure for exp {cfg.exp_id} — no data.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.6))
    handles = draw_panel(ax, df, cfg, show_ylabel=True)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="#bbbbbb",
    )
    fig.suptitle(cfg.label, fontsize=10, y=1.01)
    fig.tight_layout()

    for ext in EXT:
        path = os.path.join(OUT_DIR, f"{cfg.out_stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {OUT_DIR}{cfg.out_stem}.{{pdf,png}}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot avg-accuracy trajectories for HyperFisher experiments."
    )
    parser.add_argument(
        "--exp", nargs="+", default=["all"],
        help="Experiment IDs to plot, e.g. --exp 402 411. Use 'all' for every entry.",
    )
    args = parser.parse_args()

    if args.exp == ["all"]:
        target_ids = list(EXPERIMENT_REGISTRY.keys())
    else:
        target_ids = [int(x) for x in args.exp]

    for exp_id in target_ids:
        if exp_id not in EXPERIMENT_REGISTRY:
            print(f"[skip] exp_id={exp_id} not in EXPERIMENT_REGISTRY. "
                  f"Add it to plot this experiment.")
            continue
        plot_experiment(EXPERIMENT_REGISTRY[exp_id])

    print(f"\nAll done → {os.path.abspath(OUT_DIR)}/")