"""
Plotting script — seed-2137 initialization failure
Experiment 7: Split-MNIST Suffocated Hypernetwork (407.csv)

Produces:
  seed2137_failure_exp7.pdf / .png
"""

import ast, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

EXP_CSV     = "407.csv"
OUT_DIR     = "plots/"
VALID_SEEDS = {42, 111, 811, 1234}
FAILED_SEED = 2137

METHOD_COLORS = {
    "ifopng": "#1B6CA8","fopng": "#5BA3D9","ewc": "#2E8B57",
    "ogd": "#E07B2A","ong": "#C94040","fng": "#8B5CF6",
    "sgd": "#888888","adam": "#444444",
}
METHOD_LABELS = {
    "ifopng": "iFOPNG","fopng": "FOPNG","ewc": "EWC",
    "ogd": "OGD","ong": "ONG","fng": "FNG",
    "sgd": "SGD","adam": "Adam",
}
ORDER = ["ifopng","fopng","ogd","ong","fng","ewc","adam","sgd"]

matplotlib.rcParams.update({
    "text.usetex": False,          # set True in Overleaf run if LaTeX available
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

def safe_parse(s):
    if pd.isna(s): return {}
    try: return ast.literal_eval(str(s))
    except: return {}

def load_exp(csv_path):
    df = pd.read_csv(csv_path)
    valid_raw = defaultdict(dict)   # method -> seed -> best acc
    s2137     = {}                  # method -> acc or None (crashed)

    for _, row in df.iterrows():
        s = safe_parse(row["summary"]); c = safe_parse(row["config"])
        m = c.get("methods", ["?"]); m = m[0] if isinstance(m, list) else m
        if m not in METHOD_LABELS: continue
        seed = c.get("seed")
        acc  = s.get("best/average_accuracy")
        tc   = s.get("task_completed"); nt = c.get("num_tasks")
        completed = (tc == nt)

        if seed in VALID_SEEDS:
            if completed and acc and acc > 0.05:
                if acc > valid_raw[m].get(seed, -1):
                    valid_raw[m][seed] = acc

        elif seed == FAILED_SEED:
            if completed and acc is not None:
                if m not in s2137 or s2137[m] is None or acc > s2137[m]:
                    s2137[m] = acc
            elif m not in s2137:
                s2137[m] = None   # crashed — NaN or tc < nt

    valid = {m: list(d.values()) for m, d in valid_raw.items()}
    return valid, s2137

valid, s2137 = load_exp("results/" + EXP_CSV)

# ── debug print ──────────────────────────────────────────────────────────────
for m in ORDER:
    v = valid.get(m, [])
    v2 = s2137.get(m, "not run")
    mu = np.mean(v)*100 if v else float("nan")
    print(f"{m:8s}  valid_mean={mu:5.1f}%  n={len(v)}  seed2137={v2}")

# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.2))

n_methods = len(ORDER)
bar_w     = 0.30
gap       = 0.05
x_centers = np.arange(n_methods, dtype=float)

for i, m in enumerate(ORDER):
    col   = METHOD_COLORS[m]
    x_ctr = x_centers[i]
    x_v   = x_ctr - bar_w/2 - gap/2     # valid-seed bar centre
    x_f   = x_ctr + bar_w/2 + gap/2     # seed-2137 bar centre

    # valid-seed bar
    vals = valid.get(m, [])
    if vals:
        mu = np.mean(vals) * 100
        sd = np.std(vals, ddof=1) * 100 if len(vals) > 1 else 0.0
        ax.bar(x_v, mu, width=bar_w, color=col, alpha=0.92, zorder=3)
        if sd > 0:
            ax.errorbar(x_v, mu, yerr=sd, fmt="none", color="black",
                        capsize=3, lw=0.9, zorder=4)

    # seed-2137 bar / crash marker
    if m in s2137:
        val = s2137[m]
        if val is not None:
            ax.bar(x_f, val*100, width=bar_w, color=col, alpha=0.42,
                   hatch="///", edgecolor=col, linewidth=0.7, zorder=3)
        else:
            ax.text(x_f, 2.0, "x", ha="center", va="bottom",
                    fontsize=13, color=col, fontweight="bold", zorder=5)
    else:
        ax.text(x_f, 2.0, "-", ha="center", va="bottom",
                fontsize=9, color="#bbbbbb", zorder=5)

# degenerate reference line
ax.axhline(10.73, color="#999999", lw=0.9, ls=":", zorder=2)
ax.text(0.12, 11.5, "10.73% (degenerate)",
        ha="left", va="bottom", fontsize=7.5, color="#888888")

ax.set_xticks(x_centers)
ax.set_xticklabels([METHOD_LABELS[m] for m in ORDER])
ax.set_ylabel("Average accuracy (%)")
ax.set_ylim(0, 108)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
ax.set_axisbelow(True)

p_valid  = mpatches.Patch(facecolor="#555555", alpha=0.92,
                           label="Valid seeds {42, 111, 811, 1234} — mean ± SD")
p_failed = mpatches.Patch(facecolor="#555555", alpha=0.42, hatch="///",
                           edgecolor="#555555", lw=0.6, label="Seed 2137")
p_crash  = plt.Line2D([0],[0], marker=None, color="none",
                      label="x = crashed (NaN loss, tc < nt)")
ax.legend(handles=[p_valid, p_failed, p_crash],
          loc="upper right", framealpha=0.85, edgecolor="#cccccc")

ax.set_title("Seed-2137 initialization failure — Split-MNIST HN", pad=7)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    path = os.path.join(OUT_DIR, f"seed2137_failure_exp7.{ext}")
    fig.savefig(path)
    print(f"saved: {path}")
plt.close()