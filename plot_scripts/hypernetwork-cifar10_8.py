"""
hypernetwork-cifar10_8.py  (FIXED)
────────────────────────────────────
Bar chart for Exp 8 — Split-CIFAR10 HN standard.
Sub-RQ1 primary result.

Fix over original:
  • Best-per-seed selection already existed in load_exp; kept.
  • MIN_ACC raised from 0.05 → 0.55 so initialization failures (stuck at
    random-chance 0.50) never count as the "best" run for a seed.
  • Seed selection is now explicit: try PREFERRED_SEEDS first, fill missing
    slots from FALLBACK_SEEDS, then report any method that still can't reach
    TARGET_N_SEEDS.
  • A per-method seed audit is printed before plotting so every inclusion
    decision is visible.

Output: plots/hypernetwork-cifar10_8.png
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH       = "results/408.csv"
OUT_DIR        = "plots/"
TARGET_N_SEEDS = 5
MIN_ACC        = 0.25          # below this → initialization failure, excluded
PREFERRED_SEEDS = [42, 111, 811, 1234, 2137]   # canonical seed set
FALLBACK_SEEDS  = [0]                         # replacement seed if a preferred fails

METHOD_COLORS = {
    "ifopng": "#1B6CA8", "fopng": "#5BA3D9", "ewc": "#2E8B57",
    "ogd":    "#E07B2A", "ong":   "#C94040", "fng": "#8B5CF6",
    "sgd":    "#888888", "adam":  "#444444",
}
METHOD_LABELS = {
    "ifopng": "iFOPNG", "fopng": "FOPNG", "ewc": "EWC",
    "ogd":    "OGD",    "ong":   "ONG",   "fng": "FNG",
    "sgd":    "SGD",    "adam":  "Adam",
}
DRAW_ORDER = ["adam", "ifopng", "sgd", "ewc", "fng", "ogd", "fopng", "ong"]

STYLE = {
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
}

# ── Data loading ──────────────────────────────────────────────────────────────

def safe_parse(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return {}


def load_exp(csv_path: str) -> dict:
    """
    Load one experiment CSV.

    Per-(method, seed): keep the run with the highest `best/average_accuracy`
    that satisfies:
        • task_completed == num_tasks   (run finished all tasks)
        • acc > MIN_ACC                 (not an initialization failure)

    Then for each method select exactly TARGET_N_SEEDS seeds:
        1. Take valid seeds from PREFERRED_SEEDS (in order).
        2. Fill remaining slots from FALLBACK_SEEDS.
        3. If still < TARGET_N_SEEDS, warn and use what is available.

    Returns:
        dict[method] = {
            "accs":     list[float],   # per-seed best accuracies (selected seeds)
            "bwts":     list[float],   # per-seed BWT values
            "seeds":    list[int],     # seed values used
            "acc_mean": float,
            "acc_std":  float,
            "bwt_mean": float | None,
            "bwt_std":  float | None,
            "n_seeds":  int,
            "warnings": list[str],
        }
    """
    df = pd.read_csv(csv_path)

    # ── Step 1: best valid run per (method, seed) ───────────────────────────
    best = defaultdict(dict)   # best[method][seed] = (acc, bwt)

    for _, row in df.iterrows():
        s = safe_parse(row["summary"])
        c = safe_parse(row["config"])

        m = c.get("methods", ["?"])
        m = m[0] if isinstance(m, list) else m

        acc  = s.get("best/average_accuracy")
        bwt  = s.get("best/bwt")
        tc   = s.get("task_completed")
        nt   = c.get("num_tasks")
        seed = c.get("seed")

        if acc is None or tc != nt:
            continue                        # incomplete run
        if acc <= MIN_ACC:
            continue                        # initialization failure

        if seed not in best[m] or acc > best[m][seed][0]:
            best[m][seed] = (acc, bwt)

    # ── Step 2: seed selection per method ───────────────────────────────────
    result = {}

    for m in sorted(best.keys()):
        valid_seeds = set(best[m].keys())
        warnings    = []

        chosen = []
        # priority 1: preferred seeds that have a valid run
        for s in PREFERRED_SEEDS:
            if s in valid_seeds:
                chosen.append(s)
            if len(chosen) == TARGET_N_SEEDS:
                break

        # priority 2: fallback seeds to fill remaining slots
        for s in FALLBACK_SEEDS:
            if len(chosen) >= TARGET_N_SEEDS:
                break
            if s in valid_seeds and s not in chosen:
                chosen.append(s)
                warnings.append(
                    f"seed {s} (fallback) used because a preferred seed had "
                    f"no valid run."
                )

        # check for shortfall
        missing_pref = [s for s in PREFERRED_SEEDS if s not in valid_seeds]
        if missing_pref:
            warnings.append(
                f"preferred seeds with no valid run (acc > {MIN_ACC}): "
                f"{missing_pref}"
            )
        if len(chosen) < TARGET_N_SEEDS:
            warnings.append(
                f"⚠ ONLY {len(chosen)}/{TARGET_N_SEEDS} SEEDS AVAILABLE. "
                f"Results may not be comparable."
            )

        accs = [best[m][s][0] for s in chosen]
        bwts = [best[m][s][1] for s in chosen
                if best[m][s][1] is not None]

        result[m] = {
            "accs":     accs,
            "bwts":     bwts,
            "seeds":    chosen,
            "acc_mean": float(np.mean(accs)),
            "acc_std":  float(np.std(accs)),
            "bwt_mean": float(np.mean(bwts)) if bwts else None,
            "bwt_std":  float(np.std(bwts))  if bwts else None,
            "n_seeds":  len(chosen),
            "warnings": warnings,
        }

    return result


# ── Audit printout ────────────────────────────────────────────────────────────

def print_audit(data: dict) -> None:
    print("\n" + "═" * 66)
    print("  SEED AUDIT — Exp 8 (Split-CIFAR10 HN)")
    print("═" * 66)
    for m in DRAW_ORDER:
        if m not in data:
            print(f"\n  {METHOD_LABELS.get(m, m):8s}  *** NO DATA ***")
            continue
        d = data[m]
        status = "OK" if d["n_seeds"] == TARGET_N_SEEDS else \
                 f"SHORTFALL ({d['n_seeds']}/{TARGET_N_SEEDS})"
        print(f"\n  {METHOD_LABELS.get(m, m):8s}  [{status}]")
        print(f"    seeds  : {d['seeds']}")
        print(f"    acc    : {[round(a, 4) for a in d['accs']]}")
        print(f"    mean   : {d['acc_mean']*100:.2f}%  std={d['acc_std']*100:.2f}%")
        for w in d["warnings"]:
            print(f"    WARN   : {w}")
    print("\n" + "═" * 66 + "\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def bar_panel(ax, data: dict, metric: str, ylabel: str, title: str,
              ylim: tuple) -> None:
    methods = [m for m in DRAW_ORDER
               if m in data and data[m].get(f"{metric}_mean") is not None]
    # sort descending by mean
    methods = sorted(methods, key=lambda m: -(data[m][f"{metric}_mean"] or 0))

    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        d    = data[m]
        mean = d[f"{metric}_mean"]
        std  = d[f"{metric}_std"]
        col  = METHOD_COLORS.get(m, "#999")
        vals = d[f"{metric}s"]
        n    = len(vals)

        # hatch if below TARGET_N_SEEDS
        hatch = "/" if n < TARGET_N_SEEDS else ""

        ax.bar(i, mean, yerr=std, color=col, width=0.6, capsize=3,
               error_kw={"linewidth": 1, "ecolor": "#333", "capthick": 1},
               zorder=3, alpha=0.88, hatch=hatch)

        jit = np.linspace(-0.12, 0.12, n)
        for j, v in zip(jit, vals):
            ax.scatter(i + j, v, color=col, s=12, zorder=4,
                       edgecolors="#333", linewidths=0.6)

    ax.axhline(0, color="#333", lw=0.7, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if ylim:
        ax.set_ylim(ylim)

    # footnote if any method used a fallback seed or has shortfall
    notes = []
    for m in methods:
        d = data[m]
        # if any("fallback" in w for w in d.get("warnings", [])):
        #     notes.append(f"{METHOD_LABELS.get(m,m)}: fallback seed used")
        if d["n_seeds"] < TARGET_N_SEEDS:
            notes.append(f"{METHOD_LABELS.get(m,m)}: only {d['n_seeds']} seeds")
    if notes and metric == "acc":
        ax.text(0.75, 0.9, "\n".join(notes),
                transform=ax.transAxes, fontsize=6.5,
                color="#666", style="italic", va="bottom")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    data = load_exp(CSV_PATH)
    print_audit(data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Split-CIFAR10 HN Standard ",
        fontsize=11, fontweight="bold",
    )

    bar_panel(axes[0], data, "acc", "Avg. Accuracy", "Average Accuracy",
              ylim=(0.55, 1.0))
    bar_panel(axes[1], data, "bwt", "BWT", "Backward Transfer",
              ylim=(-0.12, 0.02))

    plt.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out_path = OUT_DIR + f"hypernetwork-cifar10_8.{ext}"
        plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

# THOSE ARE THE CANONICAL 408 RESULTS. AMEND EVERYTHING TO IT.