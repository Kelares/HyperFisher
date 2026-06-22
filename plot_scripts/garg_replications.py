"""
garg_replication_trajectories.py
────────────────────────────────
Average-accuracy trajectory figure for the five Garg-replication standalone
benchmarks. Plots all 5 benchmarks in a 2x3 grid figure.

Benchmarks (experiment_id -> task, num_tasks):
    401  permuted_mnist   (5 tasks)
    403  split_mnist_sh   (5 tasks)
    402  split_mnist_mh   (5 tasks) - *Not in original Garg et al.
    404  split_cifar10    (5 tasks)
    406  split_cifar100   (10 tasks)

Summary structure:
    summary['best/results']['acc'][str(t)] = [acc_task1, ..., acc_taskT]
        (accuracy on each task after training task t; entries for untrained
         tasks are ~0)

CI band: t-distribution confidence interval over seeds (NOT a std band).
    half_width = t_crit(conf, df=n-1) * (std / sqrt(n))
Set CONF below. Garg reports 95%; switch to 0.65 if you want the narrower band.

Output: plots/garg_repl/garg-repl-grid.pdf and .png
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from scipy import stats
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"
OUT_DIR = "plots/garg_repl/"
EXT     = ("pdf", "png")

CONF = 0.68          # confidence level for the CI band. Garg uses 0.95.
                     # set to 0.65 for the narrower band.

BENCHMARKS = {
    401: dict(name="permuted_mnist", tasks=5,  title="Permuted-MNIST",
              ylim=(0.55, 1.00), random_chance=None),
    403: dict(name="split_mnist_sh", tasks=5,  title="Split-MNIST Single-Head",
              ylim=(0.55, 1.00), random_chance=0.50),
    402: dict(name="split_mnist_mh", tasks=5,  title="Split-MNIST Multi-Head*", # <--- Star added here
              ylim=(0.77, 1.00), random_chance=0.50),
    404: dict(name="split_cifar10",  tasks=5,  title="Split-CIFAR10",
              ylim=(0.55, 1.00), random_chance=0.50),
    406: dict(name="split_cifar100", tasks=10, title="Split-CIFAR100",
              ylim=(0.05, 0.65), random_chance=0.10),
}

# ── Style: matches the bar-chart convention ──────────────────────────────────
COLORS = {
    "ifopng": "#1B6CA8",
    "fopng":  "#5BA3D9",
    "ewc":    "#2E8B57",
    "ogd":    "#E07B2A",
    "ong":    "#C94040",
    "fng":    "#8B5CF6",
    "sgd":    "#888888",
    "adam":   "#444444",
}
LABELS = {
    "ifopng": "iFOPNG",
    "fopng":  "FOPNG",
    "ewc":    "EWC",
    "ogd":    "OGD",
    "ong":    "ONG",
    "fng":    "FNG",
    "sgd":    "SGD",
    "adam":   "Adam",
}
# Draw order: baselines first, ours (iFOPNG) last so it sits on top.
METHOD_ORDER = ["sgd", "adam", "ewc", "fng", "ong", "ogd", "fopng", "ifopng"]
# iFOPNG emphasised; everything else uniform weight.
LW    = {m: (2.6 if m == "ifopng" else 1.7) for m in COLORS}
LS    = {m: "-" for m in COLORS}
LS.update({"sgd": "-", "adam": "-", "ewc": "-", "fng": "-"})
ZORDER = {m: (5 if m == "ifopng" else 3) for m in COLORS}
BAND_ALPHA = 0.13

matplotlib.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def _acc_matrix(summary: dict, num_tasks: int) -> np.ndarray | None:
    """matrix[t, i] = accuracy on task i after training task t (0-indexed t,i)."""
    results = summary.get("best/results")
    if not results or "acc" not in results:
        return None
    acc = results["acc"]
    M = np.zeros((num_tasks, num_tasks))
    for t in range(1, num_tasks + 1):
        row = acc.get(str(t)) or acc.get(t)
        if row is None:
            return None
        for i in range(t):
            v = float(row[i])
            M[t-1, i] = v if v > 1e-6 else 0.0
    return M


def _acc_trajectory(M: np.ndarray) -> np.ndarray:
    """Average accuracy over seen tasks after each task t."""
    n = M.shape[0]
    traj = np.zeros(n)
    for t in range(n):
        seen = M[t, :t+1]
        seen = seen[seen > 1e-6]
        traj[t] = seen.mean() if seen.size else 0.0
    return traj


def fetch(exp_id: int, num_tasks: int) -> dict:
    api  = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}",
                    filters={"config.experiment_id": exp_id})
    data, seen = {}, set()
    for run in runs:
        if run.state != "finished":
            continue
        method = str(run.config.get("methods", ["?"])[0]).lower().strip()
        seed   = run.config.get("seed")
        if method not in COLORS or (method, seed) in seen:
            continue
        M = _acc_matrix(dict(run.summary), num_tasks)
        if M is None:
            continue
        seen.add((method, seed))
        data.setdefault(method, []).append(_acc_trajectory(M))
    return {m: np.array(v) for m, v in data.items()}


def _ci_halfwidth(arr: np.ndarray, conf: float) -> np.ndarray:
    """t-distribution CI half-width per task. arr: (n_seeds, n_tasks)."""
    n = arr.shape[0]
    if n < 2:
        return np.zeros(arr.shape[1])
    sem   = arr.std(axis=0, ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.5 + conf / 2.0, df=n - 1)
    return tcrit * sem


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_benchmark_on_axis(ax, cfg: dict, data: dict) -> list:
    """Plots data on a specific subplot axis and returns handles for the legend."""
    num_tasks = cfg["tasks"]
    tasks = np.arange(1, num_tasks + 1)

    ax.set_title(cfg["title"], pad=6)
    ax.set_xlabel("Number of Tasks Trained")
    ax.set_ylabel("Average Accuracy (seen tasks)")
    ax.set_xlim(0.85, num_tasks + 0.15)
    ax.set_ylim(*cfg["ylim"])
    ax.set_xticks(tasks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="lightgrey", linewidth=0.5, alpha=0.6, zorder=0)

    if cfg["random_chance"] is not None:
        ax.axhline(cfg["random_chance"], color="#aaa", lw=0.8, ls=":", zorder=0)

    handles = []
    for method in METHOD_ORDER:
        if method not in data:
            continue
        arr  = data[method]                 # (n_seeds, num_tasks)
        mean = arr.mean(axis=0)
        half = _ci_halfwidth(arr, CONF)
        c, z = COLORS[method], ZORDER[method]
        ax.plot(tasks, mean, color=c, ls=LS[method], lw=LW[method], zorder=z)
        ax.fill_between(tasks, mean - half, mean + half,
                        color=c, alpha=BAND_ALPHA, zorder=z - 1)
        handles.append(mlines.Line2D([], [], color=c, ls=LS[method],
                                     lw=LW[method], label=LABELS[method]))
    
    return handles


def print_summary(cfg: dict, data: dict):
    print(f"\n=== {cfg['title']} (Exp, {cfg['tasks']} tasks) ===")
    print(f"{'Method':<8} {'n':>2}  {'final acc':>10}  {'CI half':>8}")
    for m in METHOD_ORDER:
        if m not in data:
            continue
        arr = data[m]
        final = arr[:, -1]
        half  = _ci_halfwidth(arr, CONF)[-1]
        print(f"{m:<8} {len(final):>2}  {final.mean()*100:>8.1f}%  "
              f"{half*100:>6.2f}pp")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"CI level: {int(CONF*100)}%  (t-distribution over seeds)")
    
    # Initialize 2x3 figure grid (wider figure size to accommodate 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    global_handles = None

    for idx, (exp_id, cfg) in enumerate(BENCHMARKS.items()):
        print(f"\nFetching Exp {exp_id} ({cfg['name']})...")
        data = fetch(exp_id, cfg["tasks"])
        
        ax = axes[idx]
        
        if not data:
            print(f"  WARNING: no finished runs found for Exp {exp_id}")
            ax.set_visible(False)
            continue
            
        print_summary(cfg, data)
        handles = plot_benchmark_on_axis(ax, cfg, data)
        
        # Capture handles for the global legend
        if global_handles is None and handles:
            global_handles = handles

    # Hide any unused axes (specifically the 6th one)
    for j in range(len(BENCHMARKS), len(axes)):
        axes[j].axis('off')

    # Add the unified global legend to the empty 6th subplot
    if global_handles:
        axes[-1].legend(handles=global_handles, loc="center", ncol=2,
                        frameon=False, handlelength=2.2, columnspacing=1.3,
                        fontsize=11)

    plt.tight_layout()

    pct = int(round(CONF * 100))
    
    # <--- Added caption for the multi-head experiment
    fig.text(0.99, 0.02, "* Note: Split-MNIST Multi-Head was not originally evaluated in Garg et al. (2026) [FOPNG].",
             ha="right", va="bottom", fontsize=9, color="#444")
             
    fig.text(0.99, 0.04, f"Shaded: {pct}\\% CI over seeds",
             ha="right", va="bottom", fontsize=8, color="#666")

    # Ensure the text fits by adjusting bottom margin if necessary
    fig.subplots_adjust(bottom=0.08) 

    for ext in EXT:
        out = f"{OUT_DIR}garg-repl-grid.{ext}"
        fig.savefig(out)
        print(f"  Saved: {out}")
        
    plt.close(fig)
    print("\nDone.")