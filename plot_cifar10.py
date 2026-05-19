import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import scipy.stats as stats  # <--- Added for true CI calculation

# ==============================================================================
# TYPESETTING — NeurIPS/ICML single-column style
# ==============================================================================
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "#bbbbbb",
    "figure.dpi": 300,
})

# ==============================================================================
# STYLE PALETTE
# Projection-based methods  → solid lines,   mid weight
# Regularisation / naive    → dashed/dotted, subordinate
# eFOPNG (ours)             → thick solid,   always on top
# ==============================================================================
METHOD_STYLES = {
    "EFOPNG":   {"color": "#B31B1B", "linestyle": "-",  "linewidth": 2.2,
                 "fill_alpha": 0.18, "label": "eFOPNG (Ours)"},
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

# Baselines drawn first, proposed method always rendered last (on top)
DRAW_ORDER = [m for m in METHOD_STYLES if m != "EFOPNG"] + ["EFOPNG"]

# ==============================================================================
# W&B SETUP
# ==============================================================================
api     = wandb.Api()
entity  = "michalowski-jb-tilburg-university"
project = "HyperFisher"


# ==============================================================================
# DATA FETCHER
# ==============================================================================
expected_seeds = set([42, 2137, 811, 1234, 111])

def fetch_data(job_type: str, group: str = "split_cifar10") -> pd.DataFrame:
    runs = api.runs(
        f"{entity}/{project}",
        filters={"jobType": job_type, "group": group, "state": "finished"},
    )
    print(f"[{job_type}]  {len(runs)} runs found")

    rows = []
    for run in runs:
        if run.config["seed"] not in expected_seeds:
            continue
        
        results = run.summary.get("best/results") or run.summary.get("results")
        if not results or "acc" not in results:
            mn      = run.config.get("methods", [""])[0].lower()
            results = run.summary.get(f"{mn}/results")
        if not results or "acc" not in results:
            continue

        method = run.config.get("methods")[0].upper()
        seed   = run.config.get("seed")

        for t_str, acc_list in results["acc"].items():
            t       = int(t_str)
            avg_acc = float(np.mean(acc_list[:t]))
            if avg_acc > 1.0:
                avg_acc /= 100.0
            rows.append({"Method": method, "Seed": seed, "t": t, "avg_acc": avg_acc})

    # ==============================================================================
    # CLEANED DEDUPLICATION LOGIC
    # ==============================================================================
    better = {}
    for row in rows:
        method = row["Method"]
        
        # 1. Safely initialize the method dictionary without blanking out existing keys
        if method not in better:
            better[method] = {}
        x = better[method]
        
        # 2. Key by BOTH Seed and Task 't' so your time-series line is preserved
        key = (row["Seed"], row["t"])
        if key not in x:
            x[key] = row
        else:
            # If a duplicate run exists for the same seed and task, keep the best one
            if row["avg_acc"] >= x[key]["avg_acc"]:
                x[key] = row
                
    unpacked_list = [inner_dict for seed_dict in better.values() for inner_dict in seed_dict.values()]
    print(f"[{job_type}] Processed {len(unpacked_list)} trajectory points total.")

    # ==============================================================================
    # SEED COUNT VALIDATION
    # ==============================================================================
    # 1. Gather all unique seeds across all successful runs to establish a baseline reference
    all_discovered_seeds = {key[0] for method_data in better.values() for key in method_data.keys()}
    
    # 2. If the global pool has 5+ seeds, use it as reference. Otherwise, fallback to a standard [0-4] guess.
    expected_seeds = all_discovered_seeds if len(all_discovered_seeds) >= 5 else set(range(5))
    for method, method_data in better.items():
        seeds_present = {key[0] for key in method_data.keys()}
        
        if len(seeds_present) < 5:
            missing_seeds = expected_seeds - seeds_present
            if missing_seeds:
                print(f"⚠️ [WARNING] Method '{method}' is missing seeds: {sorted(list(missing_seeds))} (Only found {len(seeds_present)}/5)")
            else:
                print(f"⚠️ [WARNING] Method '{method}' has fewer than 5 seeds. Present seeds: {sorted(list(seeds_present))}")

    return pd.DataFrame(unpacked_list)

# ==============================================================================
# SINGLE-PANEL DRAWING
# ==============================================================================
Y_MIN   = 0.70
Y_MAX   = 0.97
Y_TICKS = [0.75, 0.80, 0.85, 0.90, 0.95]
N_TASKS = 5


def draw_panel(ax: plt.Axes, df: pd.DataFrame, subtitle: str,
               show_ylabel: bool = True) -> list:
    """
    Draw one trajectory panel onto `ax`.
    Returns a list of legend handles for the shared figure legend.
    """
    ax.tick_params(direction="in", top=True, right=True, which="both")

    handles = []
    for method_name in DRAW_ORDER:
        sub = df[df["Method"] == method_name]
        if sub.empty:
            continue

        style = METHOD_STYLES.get(
            method_name,
            {"color": "#333333", "linestyle": "-", "linewidth": 1.2,
             "fill_alpha": 0.07, "label": method_name},
        )

        grouped   = sub.groupby("t")["avg_acc"]
        mean_vals = grouped.mean()
        sem_vals  = grouped.sem().fillna(0)
        n_vals    = grouped.count()  # <--- Get sample sizes (e.g., number of seeds)
        xs        = mean_vals.index.values

        # Compute the critical t-value for a 2-sided 68% CI (84th percentile).
        # np.maximum handles edge cases where n <= 1 to avoid a df=0 error.
        t_crit = stats.t.ppf(0.84, df=np.maximum(n_vals - 1, 1))
        
        # Calculate the proper margin of error
        margin = t_crit * sem_vals
        margin[n_vals <= 1] = 0.0  # No interval can be computed if only 1 sample exists

        # Clamp CI bands to the visible y-range so outlier variance
        # (e.g. SGD) does not produce bands that bleed outside the axes
        lo = np.clip(mean_vals.values - margin.values, Y_MIN, Y_MAX)
        hi = np.clip(mean_vals.values + margin.values, Y_MIN, Y_MAX)

        ax.fill_between(xs, lo, hi,
                        color=style["color"],
                        alpha=style["fill_alpha"],
                        linewidth=0)

        ax.plot(xs, mean_vals.values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"])

        handles.append(
            mlines.Line2D([], [],
                          color=style["color"],
                          linestyle=style["linestyle"],
                          linewidth=style["linewidth"],
                          label=style["label"])
        )

    # --- Axes formatting ---
    ax.set_xlabel("Number of Tasks Trained", labelpad=3)
    if show_ylabel:
        ax.set_ylabel("Accuracy on Trained Tasks", labelpad=3)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.set_xticks(range(1, N_TASKS + 1))
    ax.set_xlim(0.8, N_TASKS + 0.2)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(Y_TICKS)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("black")

    ax.set_title(subtitle, fontsize=9, pad=4)
    return handles

# ==============================================================================
# TWO-PANEL FIGURE
# ==============================================================================
def plot_cifar10_comparison(group: str = "split_cifar10"):
    df_multi = fetch_data("Target_Network",    group)
    df_hyper = fetch_data("HyperNet_Reg_True", group)  # adjust job_type as needed

    if df_multi.empty and df_hyper.empty:
        print("No data for either panel — aborting.")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                      layout="constrained")
    fig.suptitle("Split-CIFAR10", fontsize=10, y=1.02)

    handles_a = draw_panel(ax_a, df_multi, subtitle="(a) Multihead CNN",
                           show_ylabel=True)
    handles_b = draw_panel(ax_b, df_hyper, subtitle="(b) HyperNet — Single-Head CNN",
                           show_ylabel=False)

    # Shared legend below both panels — use whichever panel has more methods
    legend_handles = handles_a if len(handles_a) >= len(handles_b) else handles_b

    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.16),
        framealpha=0.95,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.5,
        fontsize=8.5,
    )
    leg.get_frame().set_linewidth(0.5)

    fig.get_layout_engine().set(hspace=0, wspace=0.05)

    os.makedirs("visualizations", exist_ok=True)
    stem = "split_cifar10_comparison"
    for ext in ("pdf", "png"):
        plt.savefig(f"visualizations/{stem}.{ext}", format=ext, bbox_inches="tight")
    plt.close()
    print(f"Saved → visualizations/{stem}.{{pdf,png}}")


# ==============================================================================
# RUN
# ==============================================================================
plot_cifar10_comparison()