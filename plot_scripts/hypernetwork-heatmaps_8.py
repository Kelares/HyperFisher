"""
hypernetwork-heatmaps_8.py — summary-based, no scan_history)
───────────────────────────────────────────────────────
All data comes from run.summary:
  {config_value}/results  →  full acc matrix  +  final BWT
  {wandb_prefix}/projection/log10_cond_A  →  condition number (last task)

Produces:
  hypernetwork-heatmaps_8-fig1_trajectory.{pdf,png}  —  avg-acc + BWT over 5 tasks, all methods
  hypernetwork-heatmaps_8-fig2_table.{pdf,png}       —  final performance table
  hypernetwork-heatmaps_8-fig3_heatmaps.{pdf,png}    —  per-task accuracy heatmaps (4 methods)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# ── Config ─────────────────────────────────────────────────────────────────
ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"
EXP_ID  = 408
OUT_DIR = "plots/"
EXT     = ["pdf", "png"]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
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

# ── Method config ──────────────────────────────────────────────────────────
# summary key for full results = f"{CONFIG_VALUE}/results"   (main.py uses CLI arg)
# summary key for cond_A       = f"{WANDB_PREFIX}/projection/log10_cond_A"
CONFIG_VALUES   = ["ifopng","fopng","ogd","ong","fng","ewc","adam","sgd"]
WANDB_PREFIXES  = ["iFOPNG","FOPNG","OGD","ONG","FNG","EWC","ADAM","SGD"]
DISPLAY_NAMES   = ["iFOPNG","FOPNG","OGD","ONG","FNG","EWC","Adam","SGD"]

CFG_TO_PREFIX  = dict(zip(CONFIG_VALUES, WANDB_PREFIXES))
CFG_TO_DISPLAY = dict(zip(CONFIG_VALUES, DISPLAY_NAMES))
DISPLAY_ORDER  = DISPLAY_NAMES

COLORS = {
    "iFOPNG":"#1f77b4","FOPNG":"#ff7f0e","OGD":"#2ca02c",
    "ONG":"#d62728",   "FNG":"#9467bd",  "EWC":"#8c564b",
    "Adam":"#e377c2",  "SGD":"#7f7f7f",
}

NUM_TASKS = 5
SEEDS     = [42, 1234, 811]

# ── Pull runs ──────────────────────────────────────────────────────────────
print(f"Connecting to '{ENTITY}/{PROJECT}'…")
api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.experiment_id": EXP_ID})
print(f"Found {len(runs)} runs with experiment_id={EXP_ID}\n")

# data[display][seed] = {"acc": {tc: [a1..a5]}, "cond_A": float|None}
data = {d: {} for d in DISPLAY_ORDER}

for run in runs:
    cfg  = run.config
    raw  = cfg.get("methods", [None])
    if isinstance(raw, list): raw = raw[0] if raw else None
    if raw is None: continue

    cfg_val  = str(raw).lower()
    if cfg_val not in CFG_TO_DISPLAY:
        print(f"  [skip] unknown config value: {raw!r}")
        continue

    display = CFG_TO_DISPLAY[cfg_val]
    prefix  = CFG_TO_PREFIX[cfg_val]
    seed    = cfg.get("seed")
    if seed not in SEEDS:
        continue

    s = run.summary

    # ── Full acc matrix from summary ───────────────────────────────────────
    results = s.get(f"{cfg_val}/results")           # main.py logs with lowercase key
    if results is None:
        results = s.get("best/results")             # fallback to best/ key

    if results is None or "acc" not in results:
        print(f"  [warn] {display} seed={seed}: no results dict in summary "
              f"(run {run.id})")
        continue

    raw_acc = results["acc"]
    # raw_acc keys are strings "1".."5", values are lists of 5 floats
    acc_dict = {}
    for tc_str, accs in raw_acc.items():
        tc = int(tc_str)
        acc_dict[tc] = [float(a) for a in accs]

    # ── Condition number (single float, last task) ─────────────────────────
    cond_A = s.get(f"{prefix}/projection/log10_cond_A")   # None for EWC/Adam/SGD

    data[display][seed] = {"acc": acc_dict, "cond_A": cond_A}
    cond_str = f"log10_cond={cond_A:.2f}" if cond_A is not None else "no cond_A"
    print(f"  {display:8s}  seed={seed}  tasks={sorted(acc_dict.keys())}  {cond_str}")

# ── Trajectory helpers ─────────────────────────────────────────────────────
def avg_acc_traj(acc_dict):
    """Mean accuracy over seen tasks at each checkpoint."""
    return np.array([
        np.nanmean(acc_dict[tc][:tc]) if tc in acc_dict else np.nan
        for tc in range(1, NUM_TASKS + 1)
    ])

def bwt_traj(acc_dict):
    """
    BWT at each checkpoint, computed directly from the acc matrix.
    BWT(t) = mean over i<t of [ acc[t][i] - acc[i][i] ]
    This needs no history scan — everything is in the acc matrix.
    """
    result = [0]   # undefined at task 1
    for tc in range(2, NUM_TASKS + 1):
        if tc not in acc_dict: result.append(np.nan); continue
        deltas = [
            acc_dict[tc][i-1] - acc_dict[i][i-1]
            for i in range(1, tc)
            if i in acc_dict
        ]
        result.append(np.mean(deltas) if deltas else np.nan)
    return np.array(result)

tasks = np.arange(1, NUM_TASKS + 1)

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Trajectory
# ═══════════════════════════════════════════════════════════════════════════
fig1, (ax_acc, ax_bwt) = plt.subplots(1, 2, figsize=(11, 4.5))

for display in DISPLAY_ORDER:
    color = COLORS[display]
    seed_accs, seed_bwts = [], []

    for sdata in data[display].values():
        ta = avg_acc_traj(sdata["acc"])
        tb = bwt_traj(sdata["acc"])       # computed from acc matrix, not stored BWT
        seed_accs.append(ta)
        seed_bwts.append(tb)
        ax_acc.plot(tasks, ta, lw=0.8, alpha=0.30, color=color)
        ax_bwt.plot(tasks, tb, lw=0.8, alpha=0.30, color=color)

    if seed_accs:
        ax_acc.plot(tasks, np.nanmean(seed_accs, 0), lw=2.2, color=color, label=display)
        ax_bwt.plot(tasks, np.nanmean(seed_bwts, 0), lw=2.2, color=color, label=display)

for ax, ylabel, title, ylim in [
    (ax_acc, "Average accuracy (seen tasks)",
     "Trajectory average accuracy\n(thin = seeds, thick = mean)", None),
    (ax_bwt, "Backward transfer (BWT)",
     "Trajectory BWT\n(thin = seeds, thick = mean)", (-0.06, 0.005)),
]:
    ax.set_xlabel("Tasks learned")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xticks(tasks)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    if ylim: ax.set_ylim(ylim)

ax_acc.legend(loc="lower left", ncol=2, fontsize=8)
fig1.suptitle(
    "Config 8 — Split-CIFAR10 Standard HN · AdamW init · Full normalisation",
    fontsize=10, y=1.01)
fig1.tight_layout()
for ext in EXT:
    p = os.path.join(OUT_DIR, f"hypernetwork-heatmaps_8-fig1_trajectory.{ext}")
    fig1.savefig(p, bbox_inches="tight", dpi=150)
print(f"\nFigure 1 → {OUT_DIR}hypernetwork-heatmaps_8-fig1_trajectory.{{pdf,png}}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Final performance table
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for display in DISPLAY_ORDER:
    seed_data = list(data[display].values())
    if not seed_data: continue

    final_accs = [avg_acc_traj(sd["acc"])[-1] for sd in seed_data]
    final_bwts = [bwt_traj(sd["acc"])[-1]     for sd in seed_data]
    final_accs = [v for v in final_accs if not np.isnan(v)]
    final_bwts = [v for v in final_bwts if not np.isnan(v)]
    if not final_accs: continue

    ma = np.mean(final_accs); sa = np.std(final_accs) if len(final_accs) > 1 else 0.
    mb = np.mean(final_bwts); sb = np.std(final_bwts) if len(final_bwts) > 1 else 0.
    rows.append((display, ma, sa, mb, sb))

if rows:
    best_acc = max(r[1] for r in rows)
    cell_text = [
        [r[0], f"{r[1]:.3f} ± {r[2]:.3f}", f"{r[3]:.3f} ± {r[4]:.3f}"]
        for r in rows
    ]
    fig2, ax2 = plt.subplots(figsize=(7, len(rows)*0.46 + 1.2))
    ax2.axis("off")
    tbl = ax2.table(cellText=cell_text,
                    colLabels=["Method", "Avg acc ± std (↑)", "BWT ± std (↑)"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for j in range(3):
        tbl[(0,j)].set_facecolor("#e0e0e0")
        tbl[(0,j)].set_text_props(fontweight="bold")
    for i, r in enumerate(rows):
        if abs(r[1] - best_acc) < 1e-9:
            for j in range(3): tbl[(i+1,j)].set_text_props(fontweight="bold")
    fig2.suptitle("Config 8 — final performance (task 5, 3 seeds)", fontsize=10, y=0.98)
    fig2.tight_layout()
    for ext in EXT:
        p = os.path.join(OUT_DIR, f"hypernetwork-heatmaps_8-fig2_table.{ext}")
        fig2.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Figure 2 → {OUT_DIR}hypernetwork-heatmaps_8-fig2_table.{{pdf,png}}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-task heatmaps
# ═══════════════════════════════════════════════════════════════════════════
HM_METHODS = [m for m in ["iFOPNG","FOPNG","EWC","Adam"] if data[m]]
if HM_METHODS:
    fig3, axes3 = plt.subplots(1, len(HM_METHODS),
                                figsize=(3.8*len(HM_METHODS), 3.6), sharey=True)
    if len(HM_METHODS) == 1: axes3 = [axes3]

    for ax, display in zip(axes3, HM_METHODS):
        matrices = []
        for sdata in data[display].values():
            mat = np.full((NUM_TASKS, NUM_TASKS), np.nan)
            for tc, accs in sdata["acc"].items():
                for te, acc in enumerate(accs):
                    if te < tc:                     # lower triangle only
                        mat[tc-1, te] = acc
            matrices.append(mat)

        with np.errstate(all="ignore"):
            mean_mat = np.nanmean(matrices, axis=0)

        mask = np.isnan(mean_mat)                   # blank upper triangle
        sns.heatmap(mean_mat, ax=ax, mask=mask,
                    vmin=0, vmax=1, annot=True, fmt=".2f",
                    annot_kws={"size": 7.5}, cmap="YlOrRd_r",
                    linewidths=0.4, linecolor="#cccccc",
                    cbar=(display == HM_METHODS[-1]))
        ax.set_title(display, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Task evaluated", fontsize=8)
        if display == HM_METHODS[0]:
            ax.set_ylabel("Tasks completed", fontsize=8)
        ax.set_xticklabels([f"T{i+1}" for i in range(NUM_TASKS)], fontsize=8)
        ax.set_yticklabels([f"T{i+1}" for i in range(NUM_TASKS)],
                           rotation=0, fontsize=8)

    fig3.suptitle(
        "Config 8 — per-task accuracy matrix (seed-averaged)\n"
        "Row = after training task T · Col = task evaluated",
        fontsize=9, y=1.02)
    fig3.tight_layout()
    for ext in EXT:
        p = os.path.join(OUT_DIR, f"hypernetwork-heatmaps_8-fig3_heatmaps.{ext}")
        fig3.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Figure 3 → {OUT_DIR}hypernetwork-heatmaps_8-fig3_heatmaps.{{pdf,png}}")

print(f"\nDone. → {os.path.abspath(OUT_DIR)}/")