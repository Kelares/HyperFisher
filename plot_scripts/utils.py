   

"""
utils.py
─────────────
Shared data-loading utilities for all thesis plot scripts.
Add to your utils.py or import directly.

Usage:
    from utils import load_exp, STYLE

    data = load_exp("results/404.csv")
    # data = {"ifopng": {"accs": [...], "bwts": [...], "n": 5,
    #                    "acc_mean": 0.81, "acc_std": 0.01,
    #                    "bwt_mean": -0.05, "bwt_std": 0.01}}
"""

import ast
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Shared style constants ────────────────────────────────────────────────

COLORS = {
    "ifopng": "#1B6CA8",
    "fopng":  "#5BA3D9",
    "ewc":    "#2E8B57",
    "ogd":    "#E07B2A",
    "ong":    "#C94040",
    "fng":    "#8B5CF6",
    "sgd":    "#888888",
    "adam":   "#444444",
}

LABELS = {
    "ifopng":     "iFOPNG",
    "ifopng_ema": "iFOPNG (EMA)",
    "fopng":      "FOPNG",
    "ewc":        "EWC",
    "ogd":        "OGD",
    "ong":        "ONG",
    "fng":        "FNG",
    "sgd":        "SGD",
    "adam":       "Adam",
}

# Canonical method display order (sorted by how interesting they are)
METHOD_ORDER = ["ifopng", "fopng", "ogd", "ong", "fng", "ewc", "adam", "sgd"]

STYLE = {
    "font.family":   "serif",
    "font.size":     9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "grid.linewidth":    0.5,
}


# ── Core loader ───────────────────────────────────────────────────────────

def _safe_parse(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return {}


def load_exp(
    fname: str,
    min_acc: float = 0.05,
    min_seeds: int = 2,
    skip_fn=None,
    acc_key: str = "best/average_accuracy",
    bwt_key: str = "best/bwt",
):
    """
    Load a WandB-exported CSV and return per-method statistics.

    Parameters
    ----------
    fname : str
        Path to the CSV file.
    min_acc : float
        Discard completed runs below this average accuracy (filters collapses).
    min_seeds : int
        Only return methods with at least this many unique seeds.
    skip_fn : callable | None
        Optional filter: skip_fn(method_str, config_dict, summary_dict) -> bool.
        Return True to discard the row.
    acc_key : str
        Summary key for average accuracy.
    bwt_key : str
        Summary key for backward transfer.

    Returns
    -------
    dict[str, dict]
        {
          method: {
            "accs":     [float, ...],   # one value per unique seed (best run)
            "bwts":     [float, ...],
            "seeds":    [seed_id, ...],
            "n":        int,
            "acc_mean": float,
            "acc_std":  float,
            "bwt_mean": float | None,
            "bwt_std":  float | None,
          }
        }
    """
    df = pd.read_csv(fname)

    # best (acc, bwt) per (method, seed)
    seed_best: dict = defaultdict(dict)

    for _, row in df.iterrows():
        s = _safe_parse(row["summary"])
        c = _safe_parse(row["config"])

        method = c.get("methods", ["?"])
        if isinstance(method, list):
            method = method[0] if method else "?"

        seed    = c.get("seed", "?")
        acc     = s.get(acc_key)
        bwt     = s.get(bwt_key)
        tc      = s.get("task_completed", "?")
        nt      = c.get("num_tasks", "?")

        if skip_fn and skip_fn(method, c, s):
            continue

        if acc is None or tc != nt or acc <= min_acc:
            continue

        # keep best-accuracy run per seed
        if seed not in seed_best[method] or acc > seed_best[method][seed][0]:
            seed_best[method][seed] = (acc, bwt)

    result = {}
    for method, seeds in seed_best.items():
        accs  = [v[0] for v in seeds.values()]
        bwts  = [v[1] for v in seeds.values() if v[1] is not None]
        seeds_list = list(seeds.keys())

        if len(accs) < min_seeds:
            continue

        result[method] = {
            "accs":     accs,
            "bwts":     bwts,
            "seeds":    seeds_list,
            "n":        len(accs),
            "acc_mean": float(np.mean(accs)),
            "acc_std":  float(np.std(accs)),
            "bwt_mean": float(np.mean(bwts)) if bwts else None,
            "bwt_std":  float(np.std(bwts))  if bwts else None,
        }

    return result


# ── Common skip functions ─────────────────────────────────────────────────

def skip_exp6(method, config, summary):
    """Exp 6 (CIFAR100 MH): keep only SGD first-task runs, drop EWC lam=50."""
    fopt = config.get("first_task_opt", "?")
    lam  = config.get("lam", "?")
    if fopt == "adam" and method not in ("sgd", "adam"):
        return True
    if method == "ewc" and lam == 50:
        return True
    return False

def skip_exp7(method, config, summary):
    if config.get("seed", "") not in [314, 111, 811, 42, 1234]:
        print(config["seed"])
        return True
    return False


def skip_exp10_contamination(method, config, summary):
    """Exp 10: only keep correct grad-only architecture (dh=64, chunk=256)."""
    dh    = config.get("hyper_hidden_dim")
    chunk = config.get("chunk_size")
    return (dh == 64 and chunk == 256)


# ── Bar chart helper ──────────────────────────────────────────────────────

def bar_panel(ax, data: dict, metric: str, ylabel: str, title: str,
              ylim=None, method_order=None):
    """
    Draw a sorted bar chart with std error bars and seed scatter dots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : dict   output of load_exp()
    metric : str  "acc" or "bwt"
    ylabel, title, ylim : as usual
    method_order : list[str] | None  — override sort order
    """
    mean_key = f"{metric}_mean"
    std_key  = f"{metric}_std"
    vals_key = f"{metric}s"

    if method_order is None:
        methods = sorted(
            [m for m in data if data[m][mean_key] is not None],
            key=lambda m: -(data[m][mean_key] or 0),
        )
    else:
        methods = [m for m in method_order if m in data
                   and data[m][mean_key] is not None]

    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        d    = data[m]
        mean = d[mean_key]
        std  = d[std_key] or 0
        col  = COLORS.get(m, "#999")
        n    = d["n"]

        ax.bar(
            i, mean, yerr=std, color=col, width=0.6, capsize=3, zorder=3,
            alpha=0.88,
            hatch="//" if n < 3 else "",   # hatch = insufficient seeds
            error_kw={"linewidth": 1, "ecolor": "#333", "capthick": 1},
        )
        pts    = d[vals_key]
        jitter = np.linspace(-0.12, 0.12, len(pts))
        for j, p in zip(jitter, pts):
            ax.scatter(i + j, p, color=col, s=12, zorder=4,
                       edgecolors="#333", linewidths=0.6)

    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS.get(m, m) for m in methods], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if ylim:
        ax.set_ylim(ylim)