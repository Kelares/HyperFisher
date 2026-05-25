"""
plot_config8.py
───────────────
Pulls Config 8 (experiment_id=408) runs from wandb and produces three
publication-ready figures:

  Figure 1  — Trajectory: avg-accuracy + BWT over 5 tasks, all 8 methods,
               thin lines = individual seeds, thick = seed mean.
  Figure 2  — Final performance summary table (mean ± std).
  Figure 3  — Per-task accuracy heatmaps for 4 key methods.

Usage:
    python plot_config8.py
    python plot_config8.py --entity YOUR_ENTITY --project HyperFisher
    python plot_config8.py --out_dir ./figures

Requirements: wandb, matplotlib, seaborn, numpy
"""

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import wandb

ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"

exp_id = 408
out_dir = "visualizations/figures_c8/"
os.makedirs(out_dir, exist_ok=True)

"""
plot_config8.py  (fixed)
────────────────────────
Pulls Config 8 (experiment_id=408) runs from wandb and produces three
publication-ready figures.

Key fixes vs v1:
  - wandb log prefixes now match the actual __name__ values:
      eFOPNG, FOPNG, OGD, ONG, FNG, EWC  (from projections / ewc)
      ADAM, SGD                            (vanilla.py uses .upper())
  - scan_history uses the correct prefix, not the raw config string
  - graceful handling of empty data (skips figures with no data)

Usage:
    python plot_config8.py
    python plot_config8.py --entity YOUR_ENTITY --project HyperFisher
    python plot_config8.py --out_dir ./figures_c8
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import wandb


os.makedirs(out_dir, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
    "legend.frameon":    False,
})

# ── Method tables ──────────────────────────────────────────────────────────
#   config_value  →  wandb_prefix  (the __name__ used in wandb.log keys)
#   projections.py: optimizer.__name__    = eFOPNG / FOPNG / OGD / ONG / FNG
#   ewc.py:         ewc.__name__          = EWC
#   vanilla.py:     optimizer_cls.__name__.upper() = ADAM / SGD

CONFIG_TO_PREFIX = {
    "efopng": "eFOPNG",
    "fopng":  "FOPNG",
    "ogd":    "OGD",
    "ong":    "ONG",
    "fng":    "FNG",
    "ewc":    "EWC",
    "adam":   "ADAM",
    "sgd":    "SGD",
}

# Display name (for legends / table rows)
PREFIX_TO_DISPLAY = {
    "eFOPNG": "eFOPNG",
    "FOPNG":  "FOPNG",
    "OGD":    "OGD",
    "ONG":    "ONG",
    "FNG":    "FNG",
    "EWC":    "EWC",
    "ADAM":   "Adam",
    "SGD":    "SGD",
}

DISPLAY_ORDER = ["eFOPNG", "FOPNG", "OGD", "ONG", "FNG", "EWC", "Adam", "SGD"]

DISPLAY_COLORS = {
    "eFOPNG": "#1f77b4",
    "FOPNG":  "#ff7f0e",
    "OGD":    "#2ca02c",
    "ONG":    "#d62728",
    "FNG":    "#9467bd",
    "EWC":    "#8c564b",
    "Adam":   "#e377c2",
    "SGD":    "#7f7f7f",
}

NUM_TASKS = 5
SEEDS = [42, 1234, 811]
EXT = ["pdf", "png"]

# ── Pull runs ──────────────────────────────────────────────────────────────
print(f"Connecting to wandb project '{PROJECT}'…")
api = wandb.Api()
prefix = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT

runs = api.runs(
    prefix,
    filters={"config.experiment_id": exp_id},
)
print(f"Found {len(runs)} runs with experiment_id={exp_id}")

# data[display_name][seed] = {"acc": {t: [a1..a5]}, "bwt": {t: float}}
data = {d: {} for d in DISPLAY_ORDER}

for run in runs:
    cfg = run.config
    raw = cfg.get("methods", [None])
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        continue

    wandb_prefix = CONFIG_TO_PREFIX.get(str(raw).lower())
    if wandb_prefix is None:
        print(f"  [skip] unknown method config value: {raw!r}")
        continue

    display = PREFIX_TO_DISPLAY[wandb_prefix]
    seed = cfg.get("seed")
    if seed not in SEEDS:
        print(f"  [skip] unexpected seed {seed} for {display}")
        continue

    # Build the exact key names the code logs
    acc_keys = [f"{wandb_prefix}/eval/acc_task_{i+1}" for i in range(NUM_TASKS)]
    bwt_key  = f"{wandb_prefix}/eval/bwt"

    history = run.scan_history(
        keys=acc_keys + [bwt_key, "task_completed"],
        page_size=500,
    )

    print(history, '\n', run.summary)

    acc_by_task, bwt_by_task = {}, {}
    for row in history:
        t = row.get("task_completed")
        if t is None:
            continue
        t = int(t)
        accs = [row.get(k) for k in acc_keys]
        if any(a is not None for a in accs):
            acc_by_task[t] = [a if a is not None else float("nan") for a in accs]
        b = row.get(bwt_key)
        if b is not None:
            bwt_by_task[t] = b

    if acc_by_task:
        data[display][seed] = {"acc": acc_by_task, "bwt": bwt_by_task}
        print(f"  {display:8s}  seed={seed}  tasks={sorted(acc_by_task.keys())}")
    else:
        print(f"  [warn] {display} seed={seed}: no acc data — "
              f"tried keys like '{acc_keys[0]}'. "
              f"Check run {run.id} in wandb for the actual key names.")

# ── Diagnostics ────────────────────────────────────────────────────────────
# If nothing loaded, print the first run's history keys to help debug
loaded = sum(len(v) for v in data.values())
if loaded == 0:
    print("\n[ERROR] No data loaded at all. Inspecting first run's available keys…")
    for run in runs[:1]:
        hist = list(run.scan_history(page_size=10))
        if hist:
            print("  Sample row keys:", list(hist[-1].keys()))
        else:
            print("  No history rows found.")
    print("\nCommon causes:")
    print("  1. The wandb prefix doesn't match — compare the keys above with CONFIG_TO_PREFIX.")
    print("  2. task_completed is not logged — check if wandb.define_metric was called.")
    print("  3. The runs belong to a different entity/project — use --entity and --project flags.")
    raise SystemExit(1)

# ── Helpers ────────────────────────────────────────────────────────────────
def avg_acc_traj(acc_dict):
    return np.array([
        np.nanmean(acc_dict[t][:t]) if t in acc_dict else float("nan")
        for t in range(1, NUM_TASKS + 1)
    ])

def bwt_traj(bwt_dict):
    return np.array([float("nan")] + [
        bwt_dict.get(t, float("nan"))
        for t in range(2, NUM_TASKS + 1)
    ])

tasks = np.arange(1, NUM_TASKS + 1)

# ═══ Figure 1: Trajectory ══════════════════════════════════════════════════
fig1, (ax_acc, ax_bwt) = plt.subplots(1, 2, figsize=(11, 4.5))

for display in DISPLAY_ORDER:
    color = DISPLAY_COLORS[display]
    seed_accs, seed_bwts = [], []
    for seed, sdata in data[display].items():
        ta = avg_acc_traj(sdata["acc"])
        tb = bwt_traj(sdata["bwt"])
        seed_accs.append(ta); seed_bwts.append(tb)
        ax_acc.plot(tasks, ta, lw=0.8, alpha=0.30, color=color)
        ax_bwt.plot(tasks, tb, lw=0.8, alpha=0.30, color=color)

    if seed_accs:
        ax_acc.plot(tasks, np.nanmean(seed_accs, 0), lw=2.2, color=color, label=display)
        ax_bwt.plot(tasks, np.nanmean(seed_bwts, 0), lw=2.2, color=color, label=display)

for ax, ylabel, title in [
    (ax_acc, "Average accuracy (seen tasks)",
     "Trajectory average accuracy\n(thin = seeds, thick = mean)"),
    (ax_bwt, "Backward transfer (BWT)",
     "Trajectory BWT\n(thin = seeds, thick = mean)"),
]:
    ax.set_xlabel("Tasks learned"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xticks(tasks)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)

ax_acc.legend(loc="lower left", ncol=2, fontsize=8)
fig1.suptitle("Config 8 — Split-CIFAR10 Standard HN · AdamW init · Full normalisation",
              fontsize=10, y=1.01)
fig1.tight_layout()

for ext in EXT:
    p = os.path.join(out_dir, f"c8_fig1_trajectory.{ext}")
    fig1.savefig(p, bbox_inches="tight", dpi=150)
    print(f"\nFigure 1 saved → {p}")

# ═══ Figure 2: Performance table ═══════════════════════════════════════════
rows, table_methods = [], []
for display in DISPLAY_ORDER:
    final_accs = [avg_acc_traj(sd["acc"])[-1] for sd in data[display].values()
                  if not np.isnan(avg_acc_traj(sd["acc"])[-1])]
    final_bwts = [bwt_traj(sd["bwt"])[-1] for sd in data[display].values()
                  if sd["bwt"] and not np.isnan(bwt_traj(sd["bwt"])[-1])]
    if not final_accs:
        continue
    table_methods.append(display)
    ma = np.mean(final_accs);  sa = np.std(final_accs) if len(final_accs) > 1 else 0.0
    mb = np.mean(final_bwts) if final_bwts else float("nan")
    sb = np.std(final_bwts)  if len(final_bwts) > 1  else 0.0
    bwt_str = f"{mb:.3f} ± {sb:.3f}" if not np.isnan(mb) else "—"
    rows.append([display, f"{ma:.3f} ± {sa:.3f}", bwt_str, ma])

if rows:
    best_acc = max(r[3] for r in rows)
    cell_text  = [[r[0], r[1], r[2]] for r in rows]
    col_labels = ["Method", "Avg acc ± std (↑)", "BWT ± std (↑)"]

    fig2, ax2 = plt.subplots(figsize=(7, len(rows) * 0.46 + 1.2))
    ax2.axis("off")
    tbl = ax2.table(cellText=cell_text, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)

    for j in range(3):                              # header row
        tbl[(0, j)].set_facecolor("#e0e0e0")
        tbl[(0, j)].set_text_props(fontweight="bold")
    for i, r in enumerate(rows):                    # bold best acc
        if abs(r[3] - best_acc) < 1e-9:
            for j in range(3):
                tbl[(i+1, j)].set_text_props(fontweight="bold")

    fig2.suptitle("Config 8 — final performance (task 5, 3 seeds)", fontsize=10, y=0.98)
    fig2.tight_layout()

    for ext in EXT:
        p = os.path.join(out_dir, f"c8_fig2_table.{ext}")
        fig2.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Figure 2 saved → {p}")
else:
    print("[warn] Figure 2 skipped — no final accuracy data available.")

# ═══ Figure 3: Per-task heatmaps ═══════════════════════════════════════════
HEATMAP_DISPLAYS = ["eFOPNG", "FOPNG", "EWC", "Adam"]
# Only keep methods that actually have data
hm_methods = [m for m in HEATMAP_DISPLAYS if data[m]]

if hm_methods:
    fig3, axes3 = plt.subplots(1, len(hm_methods),
                                figsize=(3.8 * len(hm_methods), 3.6),
                                sharey=True)
    if len(hm_methods) == 1:
        axes3 = [axes3]

    for ax, display in zip(axes3, hm_methods):
        matrices = []
        for sdata in data[display].values():
            mat = np.full((NUM_TASKS, NUM_TASKS), float("nan"))
            for tc, accs in sdata["acc"].items():
                if 1 <= tc <= NUM_TASKS:
                    for te, acc in enumerate(accs):
                        mat[tc-1, te] = acc
            matrices.append(mat)
        with np.errstate(all='ignore'):   # expected: upper-triangle cells are all-NaN
            mean_mat = np.nanmean(matrices, axis=0)

        sns.heatmap(mean_mat, ax=ax, vmin=0, vmax=1, annot=True,
                    fmt=".2f", annot_kws={"size": 7.5},
                    cmap="YlOrRd_r", linewidths=0.4, linecolor="#cccccc",
                    cbar=(display == hm_methods[-1]))
        ax.set_title(display, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Task evaluated", fontsize=8)
        if display == hm_methods[0]:
            ax.set_ylabel("Tasks completed", fontsize=8)
        ax.set_xticklabels([f"T{i+1}" for i in range(NUM_TASKS)], fontsize=8)
        ax.set_yticklabels([f"T{i+1}" for i in range(NUM_TASKS)], rotation=0, fontsize=8)

    fig3.suptitle("Config 8 — per-task accuracy matrix (seed-averaged)\n"
                  "Row = after training task T · Col = task evaluated",
                  fontsize=9, y=1.02)
    fig3.tight_layout()
    for ext in EXT:
        p = os.path.join(out_dir, f"c8_fig3_heatmaps.{ext}")
        fig3.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Figure 3 saved → {p}")
else:
    print("[warn] Figure 3 skipped — no heatmap methods have data.")

print(f"\nDone. Outputs in {os.path.abspath(out_dir)}/")