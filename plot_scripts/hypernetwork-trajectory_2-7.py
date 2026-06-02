"""
hypernetwork-trajectory_2-7.py
────────────────────────────────
Trajectory + bar chart: Exp 2 (Split-MNIST MH Standalone) vs
Exp 7 (Split-MNIST SH HN Suffocated). Sub-RQ1 MNIST comparison.
Fetches data from W&B.

Outputs:
  plots/hypernetwork-trajectory_2-7.pdf  — two-panel trajectory
  plots/hypernetwork-trajectory_2-7-bars.pdf — grouped bar + BWT
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import wandb

from utils import COLORS, LABELS

ENTITY    = "michalowski-jb-tilburg-university"
PROJECT   = "HyperFisher"
NUM_TASKS = 5
OUT       = "plots/"

# Seed 2137: initialisation failure in HN setting (acc ~ 0.10). Excluded.
EXCLUDED_SEEDS = {2137}

PANEL_META = {
    402: "(a) Multi-Head MLP — Standalone",
    407: r"(b) Suffocated Hypernetwork ($d_h\!=\!8$)",
}

matplotlib.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

METHOD_STYLE = {
    "fopng":  dict(color="#4878CF", ls="-",  lw=1.8, label="FOPNG"),
    "ong":    dict(color="#B8860B", ls="-",  lw=1.8, label="ONG"),
    "fng":    dict(color="#3CB371", ls="--", lw=1.8, label="FNG"),
    "ogd":    dict(color="#555555", ls="-",  lw=1.8, label="OGD"),
    "ewc":    dict(color="#9467bd", ls="--", lw=1.8, label="EWC"),
    "adam":   dict(color="#D62728", ls=":",  lw=1.8, label="Adam"),
    "sgd":    dict(color="#FF69B4", ls=":",  lw=1.8, label="SGD"),
    "ifopng": dict(color="#C00000", ls="-",  lw=2.6, label=r"iFOPNG (Ours)"),
}
BAND_ALPHA = 0.13
DRAW_ORDER = [m for m in METHOD_STYLE if m != "ifopng"] + ["ifopng"]


def _get_trajectory(summary, method):
    results = summary.get("best/results") or summary.get(f"{method}/results")
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


def _get_scalars(summary, method):
    acc = summary.get("best/average_accuracy") or \
          summary.get(f"{method}/eval/average_accuracy")
    bwt = (summary.get("best/results") or {}).get("bwt") or \
          summary.get("best/bwt") or summary.get(f"{method}/eval/bwt")
    if acc is None or bwt is None:
        return None
    return float(acc), float(bwt)


def fetch_all(experiment_ids):
    api   = wandb.Api()
    trajs = {eid: {} for eid in experiment_ids}
    rows  = []
    runs  = api.runs(f"{ENTITY}/{PROJECT}",
                     filters={"config.experiment_id": {"$in": experiment_ids}})
    for run in runs:
        if run.state != "finished":
            continue
        eid    = run.config.get("experiment_id")
        method = str(run.config.get("methods", "")[0]).lower().strip()
        seed   = run.config.get("seed")
        if eid not in experiment_ids or method not in METHOD_STYLE:
            continue
        if seed in EXCLUDED_SEEDS:
            continue
        s    = dict(run.summary)
        traj = _get_trajectory(s, method)
        if traj is None:
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


def plot_trajectories(trajs, save_path):
    tasks = np.arange(1, NUM_TASKS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle("Split-MNIST", fontsize=13, fontweight="bold", y=1.02)
    legend_handles = []
    for ax, (eid, panel_label) in zip(axes, PANEL_META.items()):
        ax.set_title(panel_label, pad=6)
        ax.set_xlabel("Number of Tasks Trained")
        ax.set_ylabel("Accuracy on Trained Tasks")
        ax.set_xlim(0.75, NUM_TASKS + 0.25)
        ax.set_ylim(0.40, 1.01)
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
            arr  = trajs[eid][method]
            mean = arr.mean(axis=0); std = arr.std(axis=0)
            s    = METHOD_STYLE[method]
            zord = 4 if method == "ifopng" else 2
            ax.plot(tasks, mean, color=s["color"], ls=s["ls"], lw=s["lw"], zorder=zord)
            ax.fill_between(tasks, mean - std, mean + std,
                            color=s["color"], alpha=BAND_ALPHA, zorder=zord - 1)
        if not legend_handles:
            for m in DRAW_ORDER:
                if any(m in trajs.get(e, {}) for e in PANEL_META):
                    s = METHOD_STYLE[m]
                    legend_handles.append(
                        mlines.Line2D([], [], color=s["color"], ls=s["ls"],
                                      lw=s["lw"], label=s["label"]))
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=4,
               frameon=False, handlelength=2.4, columnspacing=1.4)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_barchart(df, save_path):
    if df.empty:
        print("  [warn] No summary data")
        return
    agg = (df.groupby(["experiment_id", "method"])
             .agg(acc_mean=("final_avg_acc", "mean"),
                  acc_std =("final_avg_acc", "std"),
                  bwt_mean=("bwt", "mean"))
             .reset_index())
    hn_order  = (agg[agg.experiment_id == 407]
                 .sort_values("acc_mean", ascending=False)["method"].tolist())
    extra     = [m for m in agg[agg.experiment_id == 402]["method"].tolist()
                 if m not in hn_order]
    method_order = [m for m in hn_order + extra if m in METHOD_STYLE]
    n = len(method_order); x = np.arange(n); bw = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title(r"Split-MNIST --- Final Average Accuracy (after Task 5)", pad=8)
    ax.set_ylabel("Average Accuracy")
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_STYLE[m]["label"] for m in method_order],
                       rotation=22, ha="right")
    ax.grid(axis="y", color="lightgrey", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for eid, offset, hatch, alpha, label in [
        (402, -bw/2, "///", 0.45, "Standalone MH (Exp 2)"),
        (407,  bw/2, "",   0.85, r"Suffocated HN, $d_h\!=\!8$ (Exp 7)"),
    ]:
        sub = agg[agg.experiment_id == eid].set_index("method")
        acc_m, acc_s, bwt_m, cols = [], [], [], []
        for m in method_order:
            if m in sub.index:
                acc_m.append(sub.loc[m, "acc_mean"])
                s = sub.loc[m, "acc_std"]
                acc_s.append(float(s) if not np.isnan(s) else 0.0)
                bwt_m.append(sub.loc[m, "bwt_mean"])
            else:
                acc_m.append(0); acc_s.append(0); bwt_m.append(0)
            cols.append(METHOD_STYLE[m]["color"])
        bars = ax.bar(x + offset, acc_m, width=bw, yerr=acc_s, capsize=3,
                      color=cols, alpha=alpha, hatch=hatch, edgecolor="white",
                      linewidth=0.4, label=label, zorder=2,
                      error_kw=dict(elinewidth=0.8, ecolor="#666666"))
        for bar, bwt, err in zip(bars, bwt_m, acc_s):
            if bwt != 0 and bar.get_height() > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.012,
                        f"{bwt:+.3f}", ha="center", va="bottom",
                        fontsize=6.5, color="#333333", rotation=45)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Fetching from W&B...")
    trajs, summary_df = fetch_all([402, 407])
    plot_trajectories(trajs, OUT + "hypernetwork-trajectory_2-7.pdf")
    plot_barchart(summary_df, OUT + "hypernetwork-trajectory_2-7-bars.pdf")
    print("Done.")