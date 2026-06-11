"""
standalone_sh_mnist.py
──────────────────────
Trajectory and BWT figure for Split-MNIST Single-Head standalone (Exp 403).
All 8 methods. Two panels:
  Left  — average accuracy trajectory over 5 tasks (mean ± std band)
  Right — backward transfer trajectory over 5 tasks

Summary structure (same convention as rq1_split_mnist.py):
  summary['best/results']['acc']['t'] = [t1_acc, t2_acc, ..., tt_acc, 0, ...]
  summary['best/results']['bwt']      = scalar (final)
  summary['best/average_accuracy']    = scalar

Output: plots/standalone-sh-mnist_3.pdf
         plots/standalone-sh-mnist_3.png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import wandb

# ─────────────────────────────────────────────────────────────────────────────
ENTITY    = "michalowski-jb-tilburg-university"
PROJECT   = "HyperFisher"
EXP_ID    = 403
NUM_TASKS = 5
OUT_DIR   = "plots/"
EXT       = ("pdf", "png")

matplotlib.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

METHOD_STYLE = {
    "ifopng": dict(color="#C00000", ls="-",  lw=2.6, label=r"iFOPNG (Ours)", zorder=5),
    "fopng":  dict(color="#4878CF", ls="-",  lw=1.8, label="FOPNG",          zorder=4),
    "ewc":    dict(color="#9467bd", ls="--", lw=1.8, label="EWC",            zorder=3),
    "ogd":    dict(color="#555555", ls="-",  lw=1.8, label="OGD",            zorder=3),
    "sgd":    dict(color="#FF69B4", ls=":",  lw=1.8, label="SGD",            zorder=2),
    "adam":   dict(color="#D62728", ls=":",  lw=1.8, label="Adam",           zorder=2),
    "fng":    dict(color="#3CB371", ls="--", lw=1.8, label="FNG",            zorder=2),
    "ong":    dict(color="#B8860B", ls="-",  lw=1.8, label="ONG",            zorder=2),
}
DRAW_ORDER = [m for m in METHOD_STYLE if m != "ifopng"] + ["ifopng"]
BAND_ALPHA = 0.13


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _get_acc_matrix(summary: dict) -> np.ndarray | None:
    """
    Returns acc_matrix[t, i] = accuracy on task i after training task t.
    Shape: (NUM_TASKS, NUM_TASKS). Upper triangle is zero (untrained tasks).
    """
    results = summary.get("best/results")
    if not results or "acc" not in results:
        return None

    acc_dict = results["acc"]
    matrix = np.zeros((NUM_TASKS, NUM_TASKS))

    for t in range(1, NUM_TASKS + 1):
        row = acc_dict.get(str(t)) or acc_dict.get(t)
        if row is None:
            return None
        for i in range(t):
            matrix[t-1, i] = float(row[i]) if float(row[i]) > 1e-6 else 0.0

    return matrix


def _acc_trajectory(matrix: np.ndarray) -> np.ndarray:
    """Average accuracy after task t = mean of first t diagonal+below entries."""
    traj = np.zeros(NUM_TASKS)
    for t in range(NUM_TASKS):
        trained = matrix[t, :t+1]
        traj[t] = trained[trained > 1e-6].mean() if (trained > 1e-6).any() else 0.0
    return traj


def _bwt_trajectory(matrix: np.ndarray) -> np.ndarray:
    """
    BWT after task t = mean over i<t of (acc_matrix[t,i] - acc_matrix[i,i]).
    Returns NaN for t=0 (undefined — no prior task).
    """
    bwt = np.full(NUM_TASKS, np.nan)
    bwt[0] = 0.0   # task 1: no previous tasks, BWT = 0 by definition

    for t in range(1, NUM_TASKS):        # BWT undefined after task 1
        diffs = [matrix[t, i] - matrix[i, i] for i in range(t)]
        bwt[t] = float(np.mean(diffs)) if diffs else 0.0
    return bwt


def fetch_data() -> dict:
    api  = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.experiment_id": EXP_ID},
    )

    data = {}
    seen = {}   # (method, seed) → already loaded

    for run in runs:
        if run.state != "finished":
            continue
        method = str(run.config.get("methods", ["?"])[0]).lower().strip()
        seed   = run.config.get("seed")
        if method not in METHOD_STYLE:
            continue

        key = (method, seed)
        if key in seen:                   # keep only first (most-recent) run
            continue
        seen[key] = True

        s      = dict(run.summary)
        matrix = _get_acc_matrix(s)
        if matrix is None:
            continue

        acc_traj = _acc_trajectory(matrix)
        bwt_traj = _bwt_trajectory(matrix)

        if method not in data:
            data[method] = {"acc": [], "bwt": [], "seeds": []}
        data[method]["acc"].append(acc_traj)
        data[method]["bwt"].append(bwt_traj)
        data[method]["seeds"].append(seed)

    for m in data:
        data[m]["acc"] = np.array(data[m]["acc"])
        data[m]["bwt"] = np.array(data[m]["bwt"])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(data: dict):
    tasks = np.arange(1, NUM_TASKS + 1)

    fig, (ax_acc, ax_bwt) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle(
        r"Split-MNIST Single-Head --- Standalone Target Network (Exp~403)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ── Left: accuracy trajectory ─────────────────────────────────────────
    ax_acc.set_title("Average Accuracy (seen tasks)", pad=6)
    ax_acc.set_xlabel("Number of Tasks Trained")
    ax_acc.set_ylabel("Average Accuracy")
    ax_acc.set_xlim(0.75, NUM_TASKS + 0.25)
    ax_acc.set_ylim(0.60, 1.01)
    ax_acc.set_xticks(tasks)
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_acc.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
    ax_acc.spines[["top", "right"]].set_visible(False)
    # Mark 50% binary random chance
    ax_acc.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0,
                   label="random chance")

    # ── Right: BWT trajectory ─────────────────────────────────────────────
    ax_bwt.set_title("Backward Transfer", pad=6)
    ax_bwt.set_xlabel("Number of Tasks Trained")
    ax_bwt.set_ylabel("BWT")
    ax_bwt.set_xlim(0.75, NUM_TASKS + 0.25)
    ax_bwt.set_xticks(tasks)
    ax_bwt.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_bwt.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
    ax_bwt.axhline(0.0, color="#888", lw=0.8, ls="--", zorder=0)
    ax_bwt.spines[["top", "right"]].set_visible(False)

    legend_handles = []

    for method in DRAW_ORDER:
        if method not in data:
            continue

        s    = METHOD_STYLE[method]
        z    = s["zorder"]

        # ── accuracy ──────────────────────────────────────────────────────
        acc  = data[method]["acc"]          # (n_seeds, NUM_TASKS)
        mean = acc.mean(axis=0)
        std  = acc.std(axis=0)
        ax_acc.plot(tasks, mean,
                    color=s["color"], ls=s["ls"], lw=s["lw"], zorder=z)
        ax_acc.fill_between(tasks, mean-std, mean+std,
                            color=s["color"], alpha=BAND_ALPHA, zorder=z-1)

        # ── BWT ───────────────────────────────────────────────────────────────────
        bwt       = data[method]["bwt"]          # (n_seeds, NUM_TASKS)
        bwt_mean    = np.nanmean(bwt, axis=0)
        bwt_std     = np.nanstd (bwt, axis=0)
        bwt_mean[0] = 0.0    # definitional: no forgetting possible after task 1
        bwt_std[0]  = 0.0

        ax_bwt.plot(tasks, bwt_mean,
                    color=s["color"], ls=s["ls"], lw=s["lw"], zorder=z)
        ax_bwt.fill_between(tasks, bwt_mean - bwt_std, bwt_mean + bwt_std,
                            color=s["color"], alpha=BAND_ALPHA, zorder=z-1)
        legend_handles.append(
            mlines.Line2D([], [], color=s["color"], ls=s["ls"],
                          lw=s["lw"], label=s["label"])
        )

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), ncol=4,
               frameon=False, handlelength=2.4, columnspacing=1.4)

    plt.tight_layout()

    for ext in EXT:
        out = f"{OUT_DIR}standalone-sh-mnist_3.{ext}"
        fig.savefig(out)
        print(f"  Saved: {out}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data: dict):
    header = f"{'Method':<8}  {'n':>2}  {'Acc (final)':>13}  {'BWT (final)':>13}"
    print(f"\n{header}")
    print("─" * len(header))

    for method in DRAW_ORDER:
        if method not in data:
            continue
        acc_final = data[method]["acc"][:, -1]   # accuracy after task 5
        bwt_final = data[method]["bwt"][:, -1]   # BWT after task 5
        n = len(acc_final)
        print(f"{method:<8}  {n:>2}  "
              f"{acc_final.mean()*100:>6.1f}% ±{acc_final.std()*100:>4.1f}  "
              f"{bwt_final.mean():>+7.4f} ±{bwt_final.std():>6.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Fetching Exp {EXP_ID} from W&B ({ENTITY}/{PROJECT})...")
    data = fetch_data()

    print(f"\nSeeds loaded per method:")
    for m in DRAW_ORDER:
        if m in data:
            n = data[m]["acc"].shape[0]
            final = data[m]["acc"][:, -1].mean()
            print(f"  {m:<8}  {n} seeds  final_acc={final:.4f}")

    print_summary(data)
    print("\nPlotting...")
    plot_trajectories(data)
    print("Done.")