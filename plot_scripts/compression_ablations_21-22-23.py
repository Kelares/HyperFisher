"""
compression_ablation_421-422-423.py
────────────────────────────────────
SVD vs FIFO vs STOP gradient memory compression ablation.
20-task Permuted-MNIST, max_directions=400 (overflows at task 6).
iFOPNG with MAX Fisher accumulation throughout.

Experiments:
  421 — SVD   (compress to top-K singular vectors)
  422 — FIFO  (discard oldest task directions)
  423 — STOP  (freeze basis, discard incoming directions)

Produces:
  plots/compressionablation_421422423.png
  plots/compressionablation_421422423.pdf

Figure: two panels
  Left  — average accuracy trajectory over 20 tasks (thin=seeds, thick=mean)
          with inset over tasks 17–20
  Right — BWT trajectory from task 2 onward
  Vertical dashed line at task 5 marks first compression event.
"""

import os
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
ENTITY        = "michalowski-jb-tilburg-university"
PROJECT       = "HyperFisher"
NUM_TASKS     = 20
OUT_DIR       = "plots/"
OVERFLOW_TASK = 5   # memory fills after task 5 (5×80=400), first event at task 6

# Small x-nudge so markers don't fully stack when means coincide
MARKER_OFFSETS = {421: -0.18, 422: 0.0, 423: 0.18}
MARKER_EVERY   = 4   # one marker every N tasks on mean lines

EXP_META = {
    421: dict(label="SVD",  color="#2171b5", ls="solid",   lw=1.8, zorder=4,
              marker="o", ms=5.5, mew=0.9),
    422: dict(label="FIFO", color="#d6604d", ls="dashed",  lw=1.8, zorder=3,
              marker="s", ms=4.8, mew=0.9),
    423: dict(label="STOP", color="#41ab5d", ls="dashdot", lw=1.8, zorder=3,
              marker="^", ms=5.5, mew=0.9),
}

matplotlib.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "legend.fontsize":    9.5,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9.5,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

BAND_ALPHA = 0.14
SEED_ALPHA = 0.28
SEED_LW    = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _build_trajectory(run) -> dict | None:
    s = dict(run.summary)
    res = s.get("best/results") or s.get("ifopng/results")
    if not res or "acc" not in res:
        return None

    acc_dict = res["acc"]
    R = []
    for t in range(1, NUM_TASKS + 1):
        row = acc_dict.get(str(t)) or acc_dict.get(t)
        if row is None:
            return None
        R.append([float(v) for v in row])

    if len(R) != NUM_TASKS:
        return None

    acc_traj = np.zeros(NUM_TASKS)
    bwt_traj = np.full(NUM_TASKS, np.nan)

    for t in range(NUM_TASKS):
        seen = R[t][:t + 1]
        acc_traj[t] = np.mean(seen) if seen else 0.0
        if t > 0:
            bwt_traj[t] = np.mean([R[t][i] - R[i][i] for i in range(t)])
        else:
            bwt_traj[t] = 0.0

    return {"acc": acc_traj, "bwt": bwt_traj}


def fetch_data() -> dict:
    api  = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.experiment_id": {"$in": list(EXP_META.keys())}},
    )

    raw  = {eid: {"acc": [], "bwt": [], "seeds": []} for eid in EXP_META}
    seen = {eid: set() for eid in EXP_META}

    for run in runs:
        if run.state != "finished":
            continue
        eid  = run.config.get("experiment_id")
        seed = run.config.get("seed")
        if eid not in EXP_META:
            continue
        if seed in seen[eid]:
            print(f"  [dup]  exp={eid} seed={seed} — skipped")
            continue
        seen[eid].add(seed)

        traj = _build_trajectory(run)
        if traj is None:
            continue

        raw[eid]["acc"].append(traj["acc"])
        raw[eid]["bwt"].append(traj["bwt"])
        raw[eid]["seeds"].append(seed)
        print(f"  [ok]   exp={eid}  seed={seed}  "
              f"final_acc={traj['acc'][-1]*100:.2f}%")

    for eid in raw:
        raw[eid]["acc"] = np.array(raw[eid]["acc"])
        raw[eid]["bwt"] = np.array(raw[eid]["bwt"])

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_mean_line(ax, tasks, mean, meta, offset):
    """Draw mean line + hollow markers with x-nudge."""
    tasks_shifted = tasks + offset
    ax.plot(tasks, mean,
            color=meta["color"], ls=meta["ls"], lw=meta["lw"],
            zorder=meta["zorder"])
    ax.plot(tasks_shifted[::MARKER_EVERY], mean[::MARKER_EVERY],
            color=meta["color"], ls="none",
            marker=meta["marker"], markersize=meta["ms"],
            markeredgewidth=meta["mew"],
            markerfacecolor="white", markeredgecolor=meta["color"],
            zorder=meta["zorder"] + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot(data: dict):
    tasks = np.arange(1, NUM_TASKS + 1)

    fig, (ax_acc, ax_bwt) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle(
        r"Gradient Memory Compression Ablation --- 20-task Permuted-MNIST"
        "\n"
        r"$\texttt{iFOPNG}$ (MAX), \texttt{max\_directions}$=400$"
        r" (overflow at task~6)",
        fontsize=11, fontweight="bold", y=1.03,
    )

    # ── shared axis setup ────────────────────────────────────────────────
    for ax in (ax_acc, ax_bwt):
        ax.set_xlim(0.5, NUM_TASKS + 0.5)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(OVERFLOW_TASK + 0.5, color="#888", lw=1.0, ls="--",
                   alpha=0.7, zorder=1)

    ax_acc.set_title(r"Average Accuracy (seen tasks)", pad=5)
    ax_acc.set_xlabel("Tasks trained")
    ax_acc.set_ylabel("Average accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax_bwt.set_title(r"Backward Transfer", pad=5)
    ax_bwt.set_xlabel("Tasks trained")
    ax_bwt.set_ylabel("BWT")
    ax_bwt.axhline(0.0, color="#aaa", lw=0.8, ls=":", zorder=0)
    ax_bwt.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    legend_handles = []

    for eid, meta in EXP_META.items():
        if eid not in data or data[eid]["acc"].shape[0] == 0:
            print(f"  [warn] No data for exp {eid} ({meta['label']})")
            continue

        arr_acc = data[eid]["acc"]   # (n_seeds, NUM_TASKS)
        arr_bwt = data[eid]["bwt"]
        n       = arr_acc.shape[0]
        offset  = MARKER_OFFSETS[eid]

        # ── accuracy ─────────────────────────────────────────────────────
        mean_acc = arr_acc.mean(axis=0)
        std_acc  = arr_acc.std(axis=0)

        for seed_row in arr_acc:
            ax_acc.plot(tasks, seed_row, color=meta["color"],
                        lw=SEED_LW, alpha=SEED_ALPHA, zorder=meta["zorder"] - 1)

        _plot_mean_line(ax_acc, tasks, mean_acc, meta, offset)
        ax_acc.fill_between(tasks, mean_acc - std_acc, mean_acc + std_acc,
                            color=meta["color"], alpha=BAND_ALPHA,
                            zorder=meta["zorder"] - 1)

        # ── BWT ──────────────────────────────────────────────────────────
        mean_bwt = np.nanmean(arr_bwt, axis=0)
        std_bwt  = np.nanstd(arr_bwt, axis=0)

        for seed_row in arr_bwt:
            ax_bwt.plot(tasks, seed_row, color=meta["color"],
                        lw=SEED_LW, alpha=SEED_ALPHA, zorder=meta["zorder"] - 1)

        _plot_mean_line(ax_bwt, tasks, mean_bwt, meta, offset)
        ax_bwt.fill_between(tasks, mean_bwt - std_bwt, mean_bwt + std_bwt,
                            color=meta["color"], alpha=BAND_ALPHA,
                            zorder=meta["zorder"] - 1)

        # ── legend handle ─────────────────────────────────────────────────
        final_mean = arr_acc[:, -1].mean() * 100
        lbl = rf"{meta['label']}  ($n={n}$,  {final_mean:.1f}\%)"
        legend_handles.append(
            mlines.Line2D([], [], color=meta["color"], ls=meta["ls"],
                          lw=meta["lw"],
                          marker=meta["marker"], markersize=meta["ms"],
                          markeredgewidth=meta["mew"],
                          markerfacecolor="white",
                          markeredgecolor=meta["color"],
                          label=lbl)
        )

    # ── overflow annotation (after axes limits are finalised) ─────────────
    for ax in (ax_acc, ax_bwt):
        ylim = ax.get_ylim()
        ypos = ylim[1] - (ylim[1] - ylim[0]) * 0.03
        ax.text(OVERFLOW_TASK + 0.6, ypos,
                r"$\leftarrow$ overflow", fontsize=7.5, color="#888",
                va="top", ha="left")

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=3,
               frameon=False, handlelength=2.6, columnspacing=1.6)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = f"{OUT_DIR}compressionablation_421422423.{ext}"
        fig.savefig(out)
        print(f"  Saved: {out}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data: dict):
    header = (f"{'Strategy':<6}  {'n':>2}  "
              f"{'Final acc':>11}  {'Final BWT':>11}")
    print(f"\n{header}")
    print("─" * len(header))
    for eid, meta in EXP_META.items():
        if eid not in data or data[eid]["acc"].shape[0] == 0:
            print(f"{meta['label']:<6}  {'—':>2}")
            continue
        acc = data[eid]["acc"][:, -1]
        bwt = data[eid]["bwt"][:, -1]
        n   = len(acc)
        print(f"{meta['label']:<6}  {n:>2}  "
              f"{acc.mean()*100:>6.2f}% ±{acc.std()*100:>4.2f}  "
              f"{bwt.mean():>+7.4f} ±{bwt.std():>6.4f}")

    accs = {eid: data[eid]["acc"][:, -1].mean()
            for eid in EXP_META if data[eid]["acc"].shape[0] > 0}
    print(f"\nPairwise gaps (final acc):")
    eids = list(accs.keys())
    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            a, b = eids[i], eids[j]
            gap = (accs[a] - accs[b]) * 100
            print(f"  {EXP_META[a]['label']} vs {EXP_META[b]['label']}: {gap:+.2f} pp")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Fetching Exps 421/422/423 from W&B ({ENTITY}/{PROJECT})...")
    data = fetch_data()

    print_summary(data)
    print("\nPlotting...")
    plot(data)
    print("Done.")