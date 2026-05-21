import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import scipy.stats as stats

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
EXPECTED_SEEDS = {42, 2137, 811, 1234, 111}

def fetch_data(group: str, job_type: str = None) -> pd.DataFrame:
    # Build filter dynamically (jobType is optional if everything is in one group)
    filters = {"group": group, "state": "finished"}
    if job_type is not None:
        filters["jobType"] = job_type

    runs = api.runs(f"{entity}/{project}", filters=filters)
    print(f"[{group}] {len(runs)} runs found.")

    rows = []
    for run in runs:
        if run.config.get("seed") not in EXPECTED_SEEDS:
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

    # Deduplication
    better = {}
    for row in rows:
        method = row["Method"]
        if method not in better:
            better[method] = {}
        x = better[method]
        
        key = (row["Seed"], row["t"])
        if key not in x:
            x[key] = row
        else:
            if row["avg_acc"] >= x[key]["avg_acc"]:
                x[key] = row
                
    unpacked_list = [inner_dict for seed_dict in better.values() for inner_dict in seed_dict.values()]
    print(f"[{group}] Processed {len(unpacked_list)} trajectory points total.")

    # Validation
    all_discovered_seeds = {key[0] for method_data in better.values() for key in method_data.keys()}
    expected_seeds = all_discovered_seeds if len(all_discovered_seeds) >= 5 else set(range(5))
    
    for method, method_data in better.items():
        seeds_present = {key[0] for key in method_data.keys()}
        if len(seeds_present) < 5:
            missing_seeds = expected_seeds - seeds_present
            if missing_seeds:
                print(f"⚠️ [WARNING] Method '{method}' missing seeds: {sorted(list(missing_seeds))}")
            else:
                print(f"⚠️ [WARNING] Method '{method}' has <5 seeds. Present: {sorted(list(seeds_present))}")

    return pd.DataFrame(unpacked_list)

# ==============================================================================
# SINGLE-PANEL DRAWING
# ==============================================================================
# Widened Y-limits for "suffocated" networks which often struggle
Y_MIN   = 0.40
Y_MAX   = 1.00
Y_TICKS = [0.4, 0.6, 0.8, 1.0]
N_TASKS = 5

def draw_panel(ax: plt.Axes, df: pd.DataFrame, subtitle: str, show_ylabel: bool = True) -> list:
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
        n_vals    = grouped.count()
        xs        = mean_vals.index.values

        t_crit = stats.t.ppf(0.84, df=np.maximum(n_vals - 1, 1))
        
        margin = t_crit * sem_vals
        margin[n_vals <= 1] = 0.0

        lo = np.clip(mean_vals.values - margin.values, Y_MIN, Y_MAX)
        hi = np.clip(mean_vals.values + margin.values, Y_MIN, Y_MAX)

        ax.fill_between(xs, lo, hi, color=style["color"], alpha=style["fill_alpha"], linewidth=0)
        ax.plot(xs, mean_vals.values, color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"])

        handles.append(
            mlines.Line2D([], [], color=style["color"], linestyle=style["linestyle"],
                          linewidth=style["linewidth"], label=style["label"])
        )

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
# MAIN FIGURE
# ==============================================================================
def plot_suffocated_s_mnist(group: str = "suffocate_s_mnist_hypernetwork"):
    # Fetch without caring about job_type
    df = fetch_data(group=group, job_type=None)

    if df.empty:
        print("No data found — aborting.")
        return

    # Create a single plot layout
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2), layout="constrained")
    fig.suptitle("Split-MNIST", fontsize=10, y=1.05)

    handles = draw_panel(ax, df, subtitle="Very Low Hidden Dim Hypernetwork", show_ylabel=True)

    # Legend at the bottom
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.25),
        framealpha=0.95,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.5,
        fontsize=8.5,
    )
    leg.get_frame().set_linewidth(0.5)

    os.makedirs("visualizations", exist_ok=True)
    stem = "suffocate_s_mnist_hypernetwork"
    for ext in ("pdf", "png"):
        plt.savefig(f"visualizations/{stem}.{ext}", format=ext, bbox_inches="tight")
    plt.close()
    print(f"Saved → visualizations/{stem}.{{pdf,png}}")


# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    plot_suffocated_s_mnist()