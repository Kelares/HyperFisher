"""
Plotting script — Config 4 (Split-CIFAR10 MH Standalone, exp_id=404) vs
                  Config 8 (Split-CIFAR10 Standard HN,   exp_id=408)

Primary Sub-RQ1 comparison figure.

Produces:
  fig_rq1_cifar10_trajectory.pdf  — two-panel trajectory (main thesis figure)
  fig_rq1_cifar10_barchart.pdf    — final accuracy bar chart + BWT annotations
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ENTITY    = "michalowski-jb-tilburg-university"
PROJECT   = "HyperFisher"
NUM_TASKS = 5
EXP_IDS   = [404, 408]

EXCLUDED_SEEDS = {2137}

PANEL_META = {
    404: "(a) Multihead CNN",
    408: r"(b) HyperNet --- Single-Head CNN",
}

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
    "fopng":  dict(color="#4878CF", ls="-",  lw=1.8, label="FOPNG"),
    "ong":    dict(color="#B8860B", ls="-",  lw=1.8, label="ONG"),
    "fng":    dict(color="#3CB371", ls="--", lw=1.8, label="FNG"),
    "ogd":    dict(color="#555555", ls="-",  lw=1.8, label="OGD"),
    "ewc":    dict(color="#9467bd", ls="--", lw=1.8, label="EWC"),
    "adam":   dict(color="#D62728", ls=":",  lw=1.8, label="Adam"),
    "sgd":    dict(color="#FF69B4", ls=":",  lw=1.8, label="SGD"),
    "efopng": dict(color="#C00000", ls="-",  lw=2.6, label=r"eFOPNG (Ours)"),
}
BAND_ALPHA = 0.13
DRAW_ORDER = [m for m in METHOD_STYLE if m != "efopng"] + ["efopng"]


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching  (identical logic to Split-MNIST script)
# ─────────────────────────────────────────────────────────────────────────────

def _get_trajectory(summary: dict, method: str) -> np.ndarray | None:
    results  = summary.get("best/results") or summary.get(f"{method}/results")
    if not results or "acc" not in results:
        return None
    acc_dict = results["acc"]
    traj = []
    for t in range(1, NUM_TASKS + 1):
        acc_list = acc_dict.get(str(t)) or acc_dict.get(t)
        if acc_list is None:
            return None
        trained = [float(a) for a in acc_list[:t] if float(a) > 1e-6]
        traj.append(float(np.mean(trained)) if trained else 0.0)
    return np.array(traj)


def _get_scalars(summary: dict, method: str) -> tuple[float, float] | None:
    acc = summary.get("best/average_accuracy") or \
          summary.get(f"{method}/eval/average_accuracy")
    bwt = (summary.get("best/results") or {}).get("bwt") or \
          summary.get("best/bwt") or \
          summary.get(f"{method}/eval/bwt")
    if acc is None or bwt is None:
        return None
    return float(acc), float(bwt)


def fetch_all(experiment_ids: list[int]) -> tuple[dict, pd.DataFrame]:
    api   = wandb.Api()
    trajs = {eid: {} for eid in experiment_ids}
    rows  = []

    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.experiment_id": {"$in": experiment_ids}},
    )

    for run in runs:
        if run.state != "finished":
            print(f"  [skip] {run.name}  state={run.state}")
            continue

        eid    = run.config.get("experiment_id")
        method = str(run.config.get("methods", "")[0]).lower().strip()
        seed   = run.config.get("seed")

        if eid not in experiment_ids:
            continue
        if method not in METHOD_STYLE:
            print(f"  [skip] {run.name}  unknown method='{method}'")
            continue
        if seed in EXCLUDED_SEEDS:
            print(f"  [skip] {run.name}  excluded seed={seed}")
            continue

        s    = dict(run.summary)
        traj = _get_trajectory(s, method)
        if traj is None:
            print(f"  [warn] {run.name} ({method}, seed={seed}) — trajectory missing")
            continue

        trajs[eid].setdefault(method, []).append(traj)

        scalars = _get_scalars(s, method)
        if scalars:
            rows.append(dict(experiment_id=eid, method=method, seed=seed,
                             final_avg_acc=scalars[0], bwt=scalars[1]))

    for eid in trajs:
        for m in trajs[eid]:
            trajs[eid][m] = np.array(trajs[eid][m])

    return trajs, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Two-panel trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(trajs: dict, save_path=""):
    plot_title = "fig_rq1_cifar10_trajectory"

    tasks = np.arange(1, NUM_TASKS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle("Split-CIFAR10", fontsize=13, fontweight="bold", y=1.02)

    # Compute a shared y-floor from the data — don't hardcode
    all_finals = []
    for eid_methods in trajs.values():
        for arr in eid_methods.values():
            if arr.shape[0] > 0:
                all_finals.append(arr.mean(axis=0).min())
    y_lo = max(0.40, np.floor(min(all_finals, default=0.7) * 10) / 10 - 0.05) \
           if all_finals else 0.65

    legend_handles = []
    handles_built  = False

    for ax, (eid, panel_label) in zip(axes, PANEL_META.items()):
        ax.set_title(panel_label, pad=6)
        ax.set_xlabel("Number of Tasks Trained")
        ax.set_ylabel("Accuracy on Trained Tasks")
        ax.set_xlim(0.75, NUM_TASKS + 0.25)
        ax.set_ylim(y_lo, 1.01)
        ax.set_xticks(tasks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        if eid not in trajs or not trajs[eid]:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
            continue

        for method in DRAW_ORDER:
            if method not in trajs[eid]:
                continue
            arr   = trajs[eid][method]
            mean  = arr.mean(axis=0)
            std   = arr.std(axis=0)
            s     = METHOD_STYLE[method]
            zord  = 4 if method == "efopng" else 2

            ax.plot(tasks, mean,
                    color=s["color"], ls=s["ls"], lw=s["lw"], zorder=zord)
            ax.fill_between(tasks, mean - std, mean + std,
                            color=s["color"], alpha=BAND_ALPHA, zorder=zord - 1)

        if not handles_built:
            for m in DRAW_ORDER:
                if any(m in trajs.get(e, {}) for e in PANEL_META):
                    s = METHOD_STYLE[m]
                    legend_handles.append(
                        mlines.Line2D([], [], color=s["color"], ls=s["ls"],
                                      lw=s["lw"], label=s["label"])
                    )
            handles_built = True

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=4,
               frameon=False, handlelength=2.4, columnspacing=1.4)

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(save_path + plot_title + ext)
        print(f"  Saved: {save_path + plot_title + ext}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Bar chart with BWT annotation
# ─────────────────────────────────────────────────────────────────────────────

def plot_barchart(df: pd.DataFrame, save_path=""):
    plot_title = "fig_rq1_cifar10_barchart"

    if df.empty:
        print("  [warn] No summary data — skipping bar chart")
        return

    agg = (df.groupby(["experiment_id", "method"])
             .agg(acc_mean=("final_avg_acc", "mean"),
                  acc_std =("final_avg_acc", "std"),
                  bwt_mean=("bwt",           "mean"))
             .reset_index())

    # Sort by standalone (404) accuracy descending
    # — reveals which standalone-best methods suffer most in the HN setting
    sa_order = (agg[agg.experiment_id == 404]
                .sort_values("acc_mean", ascending=False)["method"].tolist())
    extra    = [m for m in agg[agg.experiment_id == 408]["method"].tolist()
                if m not in sa_order]
    method_order = [m for m in sa_order + extra if m in METHOD_STYLE]

    n  = len(method_order)
    x  = np.arange(n)
    bw = 0.36

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title(r"Split-CIFAR10 --- Final Average Accuracy (after Task 5)", pad=8)
    ax.set_ylabel("Average Accuracy")
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_STYLE[m]["label"] for m in method_order],
                       rotation=22, ha="right")
    ax.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, color="grey", lw=0.5, ls=":", zorder=1)

    panel_configs = [
        (404, -bw/2, "///", 0.45, "Standalone MH CNN (Config~4)"),
        (408,  bw/2, "",    0.85, "Standard HN (Config~8)"),
    ]

    for eid, offset, hatch, alpha, label in panel_configs:
        sub = agg[agg.experiment_id == eid].set_index("method")
        acc_m, acc_s, bwt_m, colors = [], [], [], []

        for m in method_order:
            if m in sub.index:
                acc_m.append(sub.loc[m, "acc_mean"])
                std_val = sub.loc[m, "acc_std"]
                acc_s.append(float(std_val) if not np.isnan(std_val) else 0.0)
                bwt_m.append(sub.loc[m, "bwt_mean"])
            else:
                acc_m.append(0); acc_s.append(0); bwt_m.append(0)
            colors.append(METHOD_STYLE[m]["color"])

        bars = ax.bar(x + offset, acc_m, width=bw,
                      yerr=acc_s, capsize=3,
                      color=colors, alpha=alpha,
                      hatch=hatch, edgecolor="white", linewidth=0.4,
                      label=label, zorder=2,
                      error_kw=dict(elinewidth=0.8, ecolor="#666666"))

        for bar, bwt, err in zip(bars, bwt_m, acc_s):
            if bwt != 0 and bar.get_height() > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.012,
                        f"{bwt:+.3f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color="#333333", rotation=45)

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.5, -0.04,
             r"\textit{Sorted by standalone accuracy. "
             r"BWT annotated above bars. "
             r"The drop from standalone to HN is the Sub-RQ1 signal.}",
             ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(save_path + plot_title + ext)
        print(f"  Saved: {save_path + plot_title + ext}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Fetching from W&B (exp_ids 404 and 408)...")
    trajs, summary_df = fetch_all(EXP_IDS)

    print("\nSeeds loaded:")
    for eid in EXP_IDS:
        print(f"  exp_id={eid}  {PANEL_META[eid]}")
        for m, arr in sorted(trajs.get(eid, {}).items()):
            print(f"    {m:10s}  {arr.shape[0]} seeds  "
                  f"final={arr[:,-1].mean():.3f} ± {arr[:,-1].std():.3f}")

    print(f"\n{len(summary_df)} summary rows loaded")

    save_path = "visualizations/"
    print("\nPlot 1 — trajectory comparison...")
    plot_trajectories(trajs, save_path)

    print("Plot 2 — bar chart with BWT...")
    plot_barchart(summary_df, save_path)

    print("\nDone.")