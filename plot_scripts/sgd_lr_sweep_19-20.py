"""
plot_sgd_threshold.py
─────────────────────
Fetches runs from wandb (state=finished) and produces a dual-panel figure:
  (a) Task-1 accuracy after 5 epochs vs SGD learning rate
  (b) Final training loss after 5 epochs vs SGD learning rate

Both panels share a log-scale x-axis. Individual seed points are shown
as grey scatter; thick line is the mean; shaded band is ±1 std.
"""

import os
import warnings
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ─────────────────────────────────────────────────────────────
ENTITY    = "michalowski-jb-tilburg-university"
PROJECT   = "HyperFisher"

# Change this to 420 for Split-CIFAR10
EXP_ID    = 419

# Experiment-specific mappings for MULTI-HEAD (2 classes per task)
EXP_CONFIG = {
    419: {
        "title": "Split-MNIST",
        "out_name": "sgd_split_mnist_lr_threshold",
        "random_chance_acc": 0.50,
        "random_chance_loss": np.log(2),
        "thresh": 1e-4  # Adjust boundary for MNIST
    },
    420: {
        "title": "Split-CIFAR10",
        "out_name": "sgd_split_cifar10_lr_threshold",
        "random_chance_acc": 0.50,
        "random_chance_loss": np.log(2),
        "thresh": 1e-2  # Adjust boundary for CIFAR10
    }
}

config = EXP_CONFIG[EXP_ID]

# ── Fetch runs ────────────────────────────────────────────────────────────────
api = wandb.Api()
print(f"Fetching runs: project={PROJECT}  experiment_id={EXP_ID}  state=finished")

runs = api.runs(
    f"{ENTITY}/{PROJECT}",
    filters={
        "config.experiment_id": EXP_ID,
        "state": "finished",
    },
)

# ── Extract metrics ───────────────────────────────────────────────────────────
records = []
for run in runs:
    lr   = run.config.get("lr") or run.config.get("first_task_lr")
    seed = run.config.get("seed")

    # Accuracy is logged at task_completed=1 via wandb.log(eval_metrics)
    acc  = run.summary.get("SGD/eval/acc_task_1")

    # Final training loss: summary holds the last logged value
    loss = run.summary.get("SGD/train/loss")

    if lr is None or acc is None:
        warnings.warn(f"  [skip] run {run.id} — missing lr ({lr}) or acc ({acc})")
        continue

    records.append({
        "lr":   float(lr),
        "seed": seed,
        "acc":  float(acc),
        "loss": float(loss) if loss is not None else float("nan"),
    })
    print(f"  lr={float(lr):.1e}  seed={seed}  acc={acc:.4f}  loss={loss}")

if not records:
    raise RuntimeError("No valid runs found. Check entity, project, and experiment_id.")

df = pd.DataFrame(records).sort_values("lr")

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(col):
    g = df.groupby("lr")[col]
    return g.agg(["mean", "std", "count"]).reset_index().sort_values("lr")

acc_stats  = agg("acc")
loss_stats = agg("loss")

# ── Styling constants ─────────────────────────────────────────────────────────
BLUE   = "#1f77b4"
RED    = "#d62728"
GREY   = "#888888"
SEED_C = "#AAAAAA"
THRESH = config["thresh"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1, figsize=(7, 6.5), sharex=True,
    gridspec_kw={"hspace": 0.08}
)

def _panel(ax, stats, df_col, ylabel, ylim, random_line=None):
    """Draw one panel: scatter + mean line + ±1 std band."""
    # Individual seeds
    ax.scatter(df["lr"], df[df_col],
               color=SEED_C, s=20, zorder=3, alpha=0.55, linewidths=0,
               label="Individual seeds")

    # Mean ± std
    ax.plot(stats["lr"], stats["mean"],
            color=BLUE, linewidth=2.2, marker="o", markersize=6,
            zorder=4, label="Mean (n seeds)")
    ax.fill_between(
        stats["lr"],
        (stats["mean"] - stats["std"]).clip(lower=0),
        stats["mean"] + stats["std"],
        alpha=0.14, color=BLUE, label="±1 std"
    )

    # Threshold line
    ax.axvline(x=THRESH, color=RED, linewidth=1.3, linestyle="--",
               alpha=0.85, zorder=5, label=f"Threshold ({THRESH:.0e})")

    # Optional horizontal reference (random chance or target loss)
    if random_line is not None:
        ax.axhline(y=random_line["y"], color=GREY, linewidth=0.9,
                   linestyle=":", alpha=0.8, label=random_line["label"])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.grid(True, which="both", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# Panel (a) — accuracy
_panel(axes[0], acc_stats, "acc",
       ylabel="Task-1 accuracy (5 epochs)",
       ylim=(-0.03, 1.05),
       random_line={"y": config["random_chance_acc"], "label": f"Random chance ({int(config['random_chance_acc']*100)}%)"})

axes[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
axes[0].set_title(
    f"SGD convergence threshold — {config['title']}\n"
    r"Task-1 accuracy and loss vs learning rate (5 epochs, $n$ seeds)",
    fontsize=12, pad=8
)

# Panel (b) — loss
max_loss = df["loss"].replace([np.inf, -np.inf], np.nan).dropna().max()
_panel(axes[1], loss_stats, "loss",
       ylabel="Final training loss (epoch 5)",
       ylim=(-0.05, min(max_loss * 1.15, 2.5)),
       random_line={"y": config["random_chance_loss"], "label": f"Random loss (ln 2 ≈ {config['random_chance_loss']:.2f})"})

axes[1].legend(fontsize=9, loc="upper right", framealpha=0.9)

# Shared x-axis
axes[1].set_xscale("log")
axes[1].set_xlabel("SGD learning rate", fontsize=11)

# Dynamically frame the x-axis based on the actual tested learning rates
min_lr = df["lr"].min()
max_lr = df["lr"].max()
axes[1].set_xlim(min_lr * 0.7, max_lr * 1.5)

# --- Force exact tick marks for all tested learning rates ---
tested_lrs = sorted(df["lr"].unique())
axes[1].set_xticks(tested_lrs)

# Format the labels neatly (e.g., 5.0e-02 -> 5e-2)
def format_lr(x):
    s = f"{x:.1e}"
    s = s.replace(".0e", "e")  # Remove trailing zero before e
    s = s.replace("e-0", "e-") # Remove leading zero in exponent
    return s

axes[1].set_xticklabels([format_lr(x) for x in tested_lrs])
axes[1].xaxis.set_minor_formatter(mticker.NullFormatter()) # Hide minor ticks to keep it clean

# Threshold annotation on both panels
for ax in axes:
    ax.annotate(
        f"  ← converges\n  ← fails",
        xy=(THRESH, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05),
        fontsize=8, color=RED, alpha=0.75,
    )

plt.tight_layout()

# Save outputs
os.makedirs("visualizations", exist_ok=True)
for ext in ("pdf", "png"):
    path = f"plots/{config['out_name']}.{ext}"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.show()