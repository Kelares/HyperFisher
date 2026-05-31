"""
plot_subrq3_adam_vs_adamw.py
────────────────────────────
Fetches Exp 5 (experiment_id=405, Adam first-task) and
Exp 8 (experiment_id=408, AdamW first-task) from WandB,
then plots the Sub-RQ3 mechanistic comparison:

  Top row   — log10(cond_A) trajectories over training (per seed)
  Bottom row — average accuracy comparison (bar, per method)

Output: plots/subrq3_adam_vs_adamw.png

Usage:
    pip install wandb matplotlib numpy
    python plot_subrq3_adam_vs_adamw.py

Set ENTITY and PROJECT below, or pass as env vars.
"""

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

# ── config ────────────────────────────────────────────────────────────────
ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"

EXP_IDS       = {405: "Adam (Exp 5)",  408: "AdamW (Exp 8)"}
METHODS       = ["ifopng", "fopng"]      # methods that log cond_A with Fisher info
TASK_STEPS    = 40                        # 200 logged steps / 5 tasks
MIN_ACC       = 0.60                      # exclude collapsed runs

os.makedirs("plots", exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

SEED_COLORS = {
    42:   "#1B6CA8",
    1234: "#E07B2A",
    2137: "#2E8B57",
    811:  "#8B5CF6",
    111:  "#C94040",
    314:  "#888888",
}
METHOD_LABELS = {"ifopng": "iFOPNG", "fopng": "FOPNG"}
TASK_BG = ["#f0f4ff", "#fff8f0", "#f0fff4", "#fff0f8", "#f8f0ff"]


# ── WandB fetch ───────────────────────────────────────────────────────────

def fetch_runs(entity, project, exp_ids):
    """
    Returns a dict: {exp_id: [run_data, ...]}

    run_data = {
        "seed":       int,
        "method":     str,
        "first_opt":  str,   # "adam" or "adamw"
        "exp_id":     int,
        "avg_acc":    float,
        "bwt":        float | None,
        "completed":  bool,
        "cond_series": {step: log10_cond_A},   # keyed by int step
    }
    """
    api = wandb.Api()

    filters = {"config.experiment_id": {"$in": list(exp_ids)}}
    runs    = api.runs(f"{entity}/{project}", filters=filters)

    result = defaultdict(list)

    for run in runs:
        cfg = run.config
        exp_id = cfg.get("experiment_id")
        if exp_id not in exp_ids:
            continue

        method = cfg.get("methods", ["?"])
        if isinstance(method, list):
            method = method[0] if method else "?"
        method = method.lower()

        if method not in METHODS:
            continue

        seed      = cfg.get("seed")
        first_opt = cfg.get("first_task_opt", "?")
        num_tasks = cfg.get("num_tasks", 5)

        summary       = run.summary._json_dict
        avg_acc       = summary.get("best/average_accuracy")
        bwt           = summary.get("best/bwt")
        task_completed = summary.get("task_completed")
        completed     = (task_completed == num_tasks)

        # Skip runs that did not finish or collapsed
        if not completed or avg_acc is None or avg_acc < MIN_ACC:
            continue

        # Extract log10_cond_A step series from summary
        # Key format: "{METHOD}/projection/log10_cond_A/{step}"
        cond_prefix = f"{method.upper() if method == 'ogd' else ('iFOPNG' if method == 'ifopng' else method.upper())}/projection/log10_cond_A/"

        # Build a normalised prefix list to handle capitalisation variants
        # e.g. "iFOPNG/projection/log10_cond_A/", "FOPNG/projection/log10_cond_A/"
        possible_prefixes = [
            f"iFOPNG/projection/log10_cond_A/",
            f"FOPNG/projection/log10_cond_A/",
            f"{method.upper()}/projection/log10_cond_A/",
            f"{method}/projection/log10_cond_A/",
        ]

        cond_series = {}
        for key, val in summary.items():
            if val is None:
                continue
            for prefix in possible_prefixes:
                if key.startswith(prefix):
                    step_str = key[len(prefix):]
                    if step_str.isdigit():
                        cond_series[int(step_str)] = val
                    break

        result[exp_id].append({
            "seed":        seed,
            "method":      method,
            "first_opt":   first_opt,
            "exp_id":      exp_id,
            "avg_acc":     avg_acc,
            "bwt":         bwt,
            "completed":   completed,
            "cond_series": cond_series,
        })
        print(f"  Fetched: exp={exp_id} method={method} seed={seed} "
              f"acc={avg_acc:.4f} cond_steps={len(cond_series)}")

    return result


# ── plotting helpers ──────────────────────────────────────────────────────

def plot_cond_panel(ax, runs, title, ymax=None):
    """Plot log10(cond_A) trajectories for a list of run_data dicts."""
    # Task background bands
    for t in range(5):
        ax.axvspan(t * TASK_STEPS + 1, (t + 1) * TASK_STEPS,
                   color=TASK_BG[t], alpha=0.35, zorder=0)
        ax.axvline((t + 1) * TASK_STEPS,
                   color="#aaa", linewidth=0.7, linestyle="--", zorder=1)
        ax.text(t * TASK_STEPS + TASK_STEPS / 2,
                (ymax or 12) * 0.96,
                f"T{t + 1}", ha="center", fontsize=7.5,
                color="#555", fontweight="bold")

    # One line per seed
    plotted_seeds = set()
    for run in runs:
        seed = run["seed"]
        series = run["cond_series"]
        if not series:
            continue
        steps = sorted(series.keys())
        vals  = [series[s] for s in steps]
        col   = SEED_COLORS.get(seed, "#888")
        label = (f"seed={seed}  (acc={run['avg_acc']:.3f})"
                 if seed not in plotted_seeds else None)
        plotted_seeds.add(seed)
        ax.plot(steps, vals, color=col, linewidth=1.8,
                label=label, zorder=3, alpha=0.9)
        # mark peak
        peak_i = int(np.argmax(vals))
        ax.scatter(steps[peak_i], vals[peak_i],
                   color=col, s=45, zorder=5,
                   marker="D", edgecolors="white", linewidths=0.8)

    # Mean ± std band across seeds
    if len(runs) > 1:
        all_steps = sorted({s for r in runs for s in r["cond_series"].keys()})
        step_means, step_stds = [], []
        valid_steps = []
        for s in all_steps:
            vals_at_s = [r["cond_series"][s] for r in runs if s in r["cond_series"]]
            if len(vals_at_s) >= 2:
                step_means.append(np.mean(vals_at_s))
                step_stds.append(np.std(vals_at_s))
                valid_steps.append(s)
        if valid_steps:
            ax.fill_between(valid_steps,
                            np.array(step_means) - np.array(step_stds),
                            np.array(step_means) + np.array(step_stds),
                            alpha=0.10, color="#555", zorder=2)

    ax.set_xlabel("Training epoch", fontsize=9)
    ax.set_ylabel("log₁₀  cond(G⊤F̂G)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(1, 200)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)


def plot_acc_panel(ax, runs_405, runs_408):
    """Side-by-side accuracy scatter per method, Adam vs AdamW."""
    method_acc = defaultdict(lambda: {"adam": [], "adamw": []})
    for r in runs_405:
        method_acc[r["method"]]["adam"].append(r["avg_acc"])
    for r in runs_408:
        method_acc[r["method"]]["adamw"].append(r["avg_acc"])

    methods = sorted(method_acc.keys())
    x       = np.arange(len(methods))
    width   = 0.3
    colors  = {"adam": "#A8C7E8", "adamw": "#1B6CA8"}
    labels  = {"adam": "Adam 1st-task (Exp 5)", "adamw": "AdamW 1st-task (Exp 8)"}

    for i, (opt, color) in enumerate([("adam", colors["adam"]),
                                       ("adamw", colors["adamw"])]):
        means = [np.mean(method_acc[m][opt]) if method_acc[m][opt] else 0
                 for m in methods]
        stds  = [np.std(method_acc[m][opt])  if method_acc[m][opt] else 0
                 for m in methods]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width, yerr=stds,
               color=color, label=labels[opt], capsize=3, zorder=3,
               alpha=0.88,
               error_kw={"linewidth": 1, "ecolor": "#333", "capthick": 1})
        for xi, (m, s) in enumerate(zip(means, stds)):
            vals = method_acc[methods[xi]][opt]
            jit  = np.linspace(-0.06, 0.06, len(vals))
            for j, v in zip(jit, vals):
                ax.scatter(xi + offset + j, v,
                           color="white", s=14, zorder=4,
                           edgecolors="#333", linewidths=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("Avg. Accuracy (5 tasks)", fontsize=9)
    ax.set_title("Accuracy: Adam vs AdamW first-task", fontsize=10,
                 fontweight="bold", pad=6)
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=8)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print(f"Fetching runs from {ENTITY}/{PROJECT} ...")
    data = fetch_runs(ENTITY, PROJECT, list(EXP_IDS.keys()))

    runs_405 = data.get(405, [])
    runs_408 = data.get(408, [])

    if not runs_405 and not runs_408:
        print("No runs fetched. Check ENTITY, PROJECT, and that "
              "experiment_id is stored in run.config.")
        return

    print(f"\nGood runs: Exp 405={len(runs_405)}, Exp 408={len(runs_408)}")

    # Separate by method for cond_A panels
    efopng_405 = [r for r in runs_405 if r["method"] == "ifopng"]
    fopng_405  = [r for r in runs_405 if r["method"] == "fopng"]
    efopng_408 = [r for r in runs_408 if r["method"] == "ifopng"]
    fopng_408  = [r for r in runs_408 if r["method"] == "fopng"]

    # Shared y-axis max across all cond panels
    all_cond_vals = [
        v for runs in [efopng_405, fopng_405, efopng_408, fopng_408]
        for r in runs for v in r["cond_series"].values()
        if v is not None
    ]
    ymax = max(all_cond_vals) * 1.08 if all_cond_vals else 12

    # ── figure: 3 rows × 2 cols ──────────────────────────────────────────
    # Row 0: iFOPNG cond_A  (Adam | AdamW)
    # Row 1: FOPNG  cond_A  (Adam | AdamW)
    # Row 2: accuracy comparison (spanning both cols)
    fig = plt.figure(figsize=(12, 11))
    fig.suptitle(
        "Sub-RQ3: Adam vs AdamW first-task — conditioning and accuracy\n"
        "Exp 5 (Adam @ 10⁻³) vs Exp 8 (AdamW @ 10⁻³), Split-CIFAR10 HyperNetwork",
        fontsize=11, fontweight="bold", y=1.01,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3,
                          height_ratios=[1, 1, 0.9])

    ax_ef_adam  = fig.add_subplot(gs[0, 0])
    ax_ef_adamw = fig.add_subplot(gs[0, 1], sharey=ax_ef_adam)
    ax_fp_adam  = fig.add_subplot(gs[1, 0], sharey=ax_ef_adam)
    ax_fp_adamw = fig.add_subplot(gs[1, 1], sharey=ax_ef_adam)
    ax_acc      = fig.add_subplot(gs[2, :])

    plot_cond_panel(ax_ef_adam,  efopng_405,
                    "iFOPNG — Adam first-task (Exp 5)", ymax=ymax)
    plot_cond_panel(ax_ef_adamw, efopng_408,
                    "iFOPNG — AdamW first-task (Exp 8)", ymax=ymax)
    plot_cond_panel(ax_fp_adam,  fopng_405,
                    "FOPNG — Adam first-task (Exp 5)", ymax=ymax)
    plot_cond_panel(ax_fp_adamw, fopng_408,
                    "FOPNG — AdamW first-task (Exp 8)", ymax=ymax)

    # Hide shared y-axis labels on right panels
    ax_ef_adamw.set_ylabel("")
    ax_fp_adamw.set_ylabel("")

    plot_acc_panel(ax_acc, runs_405, runs_408)

    # Print summary stats for thesis
    print("\n── Thesis numbers ─────────────────────────────────────────────")
    for label, runs in [("Adam 1st (Exp 5)", runs_405),
                         ("AdamW 1st (Exp 8)", runs_408)]:
        for method in METHODS:
            mrs = [r for r in runs if r["method"] == method]
            if not mrs:
                continue
            accs      = [r["avg_acc"] for r in mrs]
            max_conds = [max(r["cond_series"].values())
                         for r in mrs if r["cond_series"]]
            print(f"  {label} {method.upper()}: "
                  f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"max_log10_cond={np.mean(max_conds):.2f}±{np.std(max_conds):.2f}"
                  if max_conds else
                  f"  {label} {method.upper()}: acc={np.mean(accs):.4f}  no cond data")

    out = "plots/subrq3_adam_vs_adamw.png"
    plt.savefig(out)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()