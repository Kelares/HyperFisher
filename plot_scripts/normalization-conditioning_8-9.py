import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

try:
    from utils import STYLE
except ImportError:
    STYLE = {}

# --- Configuration ---
ENTITY  = "michalowski-jb-tilburg-university"
PROJECT = "HyperFisher"
RESULTS = "results/"
OUT     = "plots/"

TASK_MAX_STEPS = 50  # Hard limit of epochs per task
TASK_BG = ["#fff8f0", "#f0fff4", "#fff0f8", "#f8f0ff"] 
SEED_COLORS = {42: "#1B6CA8", 1234: "#E07B2A", 811: "#2E8B57",
               2137: "#8B5CF6", 111: "#C94040"}

def fetch_task_lengths(exp_ids):
    """Fetches exact epoch counts for tasks 2-5 directly from WandB, ignoring metric prefixes."""
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}"
    known_lengths = {}

    for exp_id in exp_ids:
        print(f"\nFetching W&B task lengths for Experiment {exp_id}...")
        
        filters = {
            "$or": [
                {"config.experiment_id": exp_id},
                {"config.experiment_id": str(exp_id)}
            ]
        }
        
        runs = api.runs(path, filters=filters)
        exp_lengths = {}
        best_accs = {} 
        
        for run in runs:
            seed = run.config.get("seed")
            if seed is None: continue
                
            run_acc = run.summary.get("best/average_accuracy")
            if run_acc is None:
                best_dict = run.summary.get("best")
                if isinstance(best_dict, dict):
                    run_acc = best_dict.get("average_accuracy")
            if run_acc is None:
                acc_key = next((k for k in run.summary.keys() if "eval/average_accuracy" in k.lower()), None)
                if acc_key: run_acc = run.summary.get(acc_key)
            if run_acc is None: continue
                
            if seed not in best_accs or run_acc > best_accs[seed]:
                history = run.history(samples=5000)
                if history.empty or 'task' not in history.columns: continue
                    
                actual_loss_col = next((col for col in history.columns if col.lower().endswith("train/loss")), None)
                if actual_loss_col is None: continue
                    
                lengths = []
                for t_idx in [2, 3, 4, 5]:
                    valid_steps = history[(history['task'] == t_idx) & (history[actual_loss_col].notnull())]
                    lengths.append(len(valid_steps))
                    
                best_accs[seed] = run_acc
                exp_lengths[seed] = lengths
            
        known_lengths[exp_id] = exp_lengths
        
    return known_lengths


def extract_cond_series(fname, method="ifopng", min_acc=0.6):
    df = pd.read_csv(fname)
    by_seed = {}
    
    for _, row in df.iterrows():
        try:
            s = ast.literal_eval(str(row["summary"]))
            c = ast.literal_eval(str(row["config"]))
        except Exception: continue
            
        m = c.get("methods", ["?"])
        if isinstance(m, list): m = m[0]
        if m.lower() != method: continue
        
        tc = s.get("task_completed", "?")
        nt = c.get("num_tasks", "?")
        if tc != nt: continue
        
        avg_acc = s.get("best/average_accuracy")
        if avg_acc is None or avg_acc < min_acc: continue

        seed = c.get("seed")
        series = {}
        for k, v in s.items():
            if "log10_cond_A/" in k and v is not None:
                try:
                    step = int(k.rsplit("/", 1)[-1])
                    series[step] = v
                except ValueError:
                    continue
        if not series: continue

        if seed not in by_seed or avg_acc > by_seed[seed]["avg_acc"]:
            by_seed[seed] = {"seed": seed, "series": series, "avg_acc": avg_acc}

    return list(by_seed.values())


def _draw_panel(ax, runs, title, ymax, marker, exp_task_lengths, annotate_explosion=False):
    if not runs: return

    # 1. Draw static background grid (Always 50 steps per task)
    for t in range(4):
        start_x = t * TASK_MAX_STEPS + 1
        end_x = (t + 1) * TASK_MAX_STEPS
        ax.axvspan(start_x, end_x, color=TASK_BG[t % len(TASK_BG)], alpha=0.35, zorder=0)
        ax.axvline(end_x, color="#aaa", linewidth=0.7, linestyle="--", zorder=1)
        ax.text(start_x + TASK_MAX_STEPS / 2 - 0.5, ymax * 0.96, f"T{t+2}", 
                ha="center", fontsize=7.5, color="#555", fontweight="bold")

    all_mapped_runs = []
    
    # 2. Process and map each run into the fixed windows
    for run in runs:
        seed  = run["seed"]
        steps = sorted(run["series"].keys())
        vals  = [run["series"][s] for s in steps]
        
        # Get W&B exact epoch lengths (fallback to 50 if missing)
        lengths = exp_task_lengths.get(seed, [TASK_MAX_STEPS] * 4)
        
        x_mapped = []
        y_mapped = []
        mapped_dict = {}
        idx = 0
        
        for t_idx, t_len in enumerate(lengths):
            if t_len == 0 or idx >= len(vals): continue
            
            # The start of this task's fixed 50-step window (0, 50, 100, 150)
            window_start = t_idx * TASK_MAX_STEPS
            
            for i in range(t_len):
                if idx >= len(vals): break
                current_x = window_start + i + 1
                
                x_mapped.append(current_x)
                y_mapped.append(vals[idx])
                mapped_dict[current_x] = vals[idx]
                idx += 1
                
            # ADD A NaN BREAK to visually disconnect the line before the next task begins
            x_mapped.append(window_start + t_len + 1)
            y_mapped.append(np.nan)

        # Plot the broken line
        col = SEED_COLORS.get(seed, "#888")
        lbl = f"seed={seed}  (acc={run['avg_acc']:.3f})"
        
        ax.plot(x_mapped, y_mapped, color=col, linewidth=1.8, label=lbl, zorder=3, alpha=0.9)
        
        # Mark peak and optionally annotate
        valid_y = [y for y in y_mapped if not np.isnan(y)]
        if valid_y:
            peak_val = max(valid_y)
            peak_x = x_mapped[y_mapped.index(peak_val)]
            ax.scatter(peak_x, peak_val, color=col, s=45, zorder=5, marker=marker, edgecolors="white", linewidths=0.8)
            
            if annotate_explosion and peak_val > 7:
                ax.annotate(
                    f"cond ~ 10^{peak_val:.1f}",
                    xy=(peak_x, peak_val),
                    xytext=(peak_x - 30, peak_val - 1.8),
                    fontsize=7, color=col,
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=col),
                )
                
        all_mapped_runs.append({"series": mapped_dict})

    # 3. Calculate accurate bands across aligned task epochs
    if len(all_mapped_runs) > 1:
        all_steps = sorted({s for r in all_mapped_runs for s in r["series"]})
        vmeans, vstds, vsteps = [], [], []
        for s in all_steps:
            vs = [r["series"][s] for r in all_mapped_runs if s in r["series"]]
            if len(vs) >= 2:
                vmeans.append(np.mean(vs))
                vstds.append(np.std(vs))
                vsteps.append(s)
                
        # Draw the fill_between separately for each task so gaps don't draw weird connecting polygons
        for t in range(4):
            start_x = t * TASK_MAX_STEPS + 1
            end_x = (t + 1) * TASK_MAX_STEPS
            
            segment_x = [x for x in vsteps if start_x <= x <= end_x]
            segment_mean = [vmeans[vsteps.index(x)] for x in segment_x]
            segment_std = [vstds[vsteps.index(x)] for x in segment_x]
            
            if segment_x:
                ax.fill_between(segment_x,
                                np.array(segment_mean) - np.array(segment_std),
                                np.array(segment_mean) + np.array(segment_std),
                                alpha=0.10, color="#555", zorder=2)
                            
    ax.set_xlabel("Task Epoch Windows (Early Stopping Indicated by Gaps)", fontsize=9)
    ax.set_ylabel("log₁₀ cond(G⊤F̂G)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    
    ax.set_xlim(1, 4 * TASK_MAX_STEPS + 5) 
    ax.set_ylim(0, ymax)
    # Positions the legend below the X-axis in a clean 2-column grid
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), 
              ncol=2, framealpha=0.9, borderaxespad=0.)

def main():
    os.makedirs(OUT, exist_ok=True)
    if STYLE: plt.rcParams.update(STYLE)

    print("--- 1. Syncing dynamic boundaries from W&B (Best Accuracy Match) ---")
    dynamic_lengths = fetch_task_lengths([408, 409])

    print("\n--- 2. Parsing local CSV conditions ---")
    runs_8 = extract_cond_series(RESULTS + "408.csv", min_acc=0.6)
    runs_9 = extract_cond_series(RESULTS + "409.csv", min_acc=0.0)

    all_vals = [v for r in runs_8 + runs_9 for v in r["series"].values()]
    ymax     = max(all_vals) * 1.08 if all_vals else 12

    print("\n--- 3. Rendering plots ---")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle(
        "Sub-RQ2: Projection Matrix Conditioning — iFOPNG on Split-CIFAR10 HN\n"
        "Full normalization (Exp 8) vs No normalization (Exp 9)",
        fontsize=10, fontweight="bold",
    )

    _draw_panel(axes[0], runs_8, "With Normalization (Exp 8)", ymax, marker="o", 
                exp_task_lengths=dynamic_lengths.get(408, {}))
    
    _draw_panel(axes[1], runs_9, "Without Normalization (Exp 9)", ymax, marker="D", 
                exp_task_lengths=dynamic_lengths.get(409, {}), annotate_explosion=True)

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "normalization-conditioning_8-9.png", bbox_inches="tight")
    plt.close()
    print("Done! Saved to", OUT + "normalization-conditioning_8-9.png")

if __name__ == "__main__":
    main()