import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("plots", exist_ok=True)
plt.rcParams.update({
    "font.family":"serif","font.size":9,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.8,"figure.dpi":150,
    "savefig.dpi":200,"savefig.bbox":"tight",
    "axes.grid":True,"grid.alpha":0.2,"grid.linewidth":0.5,
})

# ── Data: iFOPNG vs FOPNG across all standalone + HN benchmarks ────────────
# Format: (label, efopng_accs, fopng_accs, efopng_bwts, fopng_bwts)
# accs/bwts = list of per-seed values for mean ± std

import pandas as pd, ast
from collections import defaultdict

def safe_parse(s):
    if pd.isna(s): return {}
    try: return ast.literal_eval(str(s))
    except: return {}

def get_ef_fp(fname, min_acc=0.05, skip_fn=None):
    """Returns {method: [acc per seed (best per seed)]}"""
    df = pd.read_csv(f"results/{fname}")
    seed_best = defaultdict(lambda: defaultdict(lambda: -999))
    seed_bwt  = defaultdict(lambda: defaultdict(lambda: None))
    for _, row in df.iterrows():
        s = safe_parse(row["summary"]); c = safe_parse(row["config"])
        m = c.get("methods",["?"])
        if isinstance(m,list): m=m[0]
        if m not in ("ifopng","fopng"): continue
        acc = s.get("best/average_accuracy",None)
        bwt = s.get("best/bwt",None)
        tc  = s.get("task_completed","?"); nt = c.get("num_tasks","?")
        seed = c.get("seed","?")
        if skip_fn and skip_fn(m,c,s): continue
        if acc and tc==nt and acc>min_acc:
            if acc > seed_best[m][seed]:
                seed_best[m][seed] = acc
                seed_bwt[m][seed]  = bwt
    result = {}
    for m in ("ifopng","fopng"):
        if m in seed_best and len(seed_best[m]) >= 2:
            result[m] = {
                "accs": list(seed_best[m].values()),
                "bwts": [v for v in seed_bwt[m].values() if v is not None]
            }
    return result

# Load each experiment
def skip406(m,c,s):
    return c.get("first_task_opt")=="adam" and m not in ("sgd","adam") or \
           m=="ewc" and c.get("lam")==50

benchmarks = [
    ("Perm-MNIST\n(Exp 1)",   get_ef_fp("401.csv"), "standalone"),
    ("MNIST MH\n(Exp 2)",     get_ef_fp("402.csv"), "standalone"),
    ("CIFAR10 MH\n(Exp 4)",   get_ef_fp("404.csv"), "standalone"),
    ("CIFAR100 MH\n(Exp 6)",  get_ef_fp("406.csv", skip_fn=skip406), "standalone"),
    ("MNIST HN\n(Exp 7)",     get_ef_fp("407.csv"), "HN"),
    ("CIFAR10 HN\n(Exp 8)",   get_ef_fp("408.csv"), "HN"),
]

# ── Figure: 2 panels ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Sub-RQ3: iFOPNG vs FOPNG — Does the elastic Fisher preconditioner help?\n"
    "Across standalone benchmarks (left) and hypernetwork benchmarks (right of divider)",
    fontsize=10, fontweight="bold"
)

COL_EF = "#1B6CA8"
COL_FP = "#5BA3D9"
WIDTH   = 0.32
x       = np.arange(len(benchmarks))

# ── Left panel: grouped bars iFOPNG vs FOPNG ─────────────────────────────
ax = axes[0]
for xi, (label, data, btype) in enumerate(benchmarks):
    for offset, key, col in [(- WIDTH/2, "ifopng", COL_EF),
                              (+ WIDTH/2, "fopng",  COL_FP)]:
        if key not in data: continue
        accs = data[key]["accs"]
        mean, std = np.mean(accs), np.std(accs)
        ax.bar(xi+offset, mean, width=WIDTH, color=col,
               yerr=std, capsize=3, zorder=3, alpha=0.88,
               error_kw={"linewidth":0.8,"ecolor":"#333","capthick":0.8})
        jit = np.linspace(-0.05, 0.05, len(accs))
        for j, v in zip(jit, accs):
            ax.scatter(xi+offset+j, v, color="white", s=10,
                       zorder=4, edgecolors="#333", linewidths=0.5)

# Divider between standalone and HN
ax.axvline(3.5, color="#999", lw=1.0, ls="--", alpha=0.6)
ax.text(3.55, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
        "HN →", fontsize=7.5, color="#666", va="top")

ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in benchmarks], fontsize=7.5)
ax.set_ylabel("Avg. Accuracy", fontsize=9)
ax.set_ylim(0.20, 1.05)
ax.set_title("Accuracy: iFOPNG vs FOPNG per benchmark", fontsize=9,
             fontweight="bold", pad=6)
ax.legend(handles=[
    mpatches.Patch(color=COL_EF, label="iFOPNG (elastic F_c = F_new + F_old)"),
    mpatches.Patch(color=COL_FP, label="FOPNG  (rigid projection)"),
], fontsize=7.5, loc="lower left")

# ── Right panel: Δ = iFOPNG − FOPNG ─────────────────────────────────────
ax2 = axes[1]
deltas, delta_stds, colors_bar = [], [], []
for label, data, btype in benchmarks:
    if "ifopng" in data and "fopng" in data:
        ef = np.mean(data["ifopng"]["accs"])
        fp = np.mean(data["fopng"]["accs"])
        # Approximate std of difference (independent seeds)
        ef_s = np.std(data["ifopng"]["accs"])
        fp_s = np.std(data["fopng"]["accs"])
        deltas.append(ef - fp)
        delta_stds.append(np.sqrt(ef_s**2 + fp_s**2))
        colors_bar.append("#2E8B57" if (ef - fp) > 0.01 else "#888888")
    else:
        deltas.append(0); delta_stds.append(0); colors_bar.append("#ccc")

bars = ax2.bar(x, deltas, yerr=delta_stds, color=colors_bar, width=0.55,
               capsize=4, zorder=3, alpha=0.88,
               error_kw={"linewidth":1,"ecolor":"#333","capthick":1})

# Annotate delta values
for xi, (d, std) in enumerate(zip(deltas, delta_stds)):
    ax2.text(xi, d + std + 0.005, f"+{d:.3f}" if d >= 0 else f"{d:.3f}",
             ha="center", fontsize=7.5, fontweight="bold",
             color="#2E8B57" if d > 0.01 else "#666")

ax2.axhline(0, color="#333", lw=1.0, ls="--", alpha=0.6)
ax2.axhline(0.01, color="#aaa", lw=0.7, ls=":", alpha=0.5)
ax2.text(-0.45, 0.012, "trivial threshold (1pp)", fontsize=7,
         color="#aaa", va="bottom")
ax2.axvline(3.5, color="#999", lw=1.0, ls="--", alpha=0.6)
ax2.text(3.55, max(deltas)*1.1, "HN →", fontsize=7.5, color="#666")

ax2.set_xticks(x)
ax2.set_xticklabels([b[0] for b in benchmarks], fontsize=7.5)
ax2.set_ylabel("Δ Accuracy  (iFOPNG − FOPNG)", fontsize=9)
ax2.set_title("iFOPNG advantage over FOPNG", fontsize=9,
              fontweight="bold", pad=6)
ymax = max(deltas) + max(delta_stds) + 0.02
ax2.set_ylim(-0.02, ymax * 1.15)

# Shade region of substantial advantage
ax2.axhspan(0.05, ymax*1.15, alpha=0.04, color="#2E8B57", zorder=0)
ax2.text(5.45, 0.055, "substantial\n(>5pp)", fontsize=6.5,
         color="#2E8B57", ha="right", va="bottom")

plt.tight_layout(pad=1.5)
plt.savefig("plots/subrq3_efopng_summary.png")
plt.close()

# Print thesis numbers
print("iFOPNG vs FOPNG summary:")
print(f"{'Benchmark':<20} {'iFOPNG':>8} {'FOPNG':>8} {'Delta':>8}")
for label, data, btype in benchmarks:
    if "ifopng" in data and "fopng" in data:
        ef = np.mean(data["ifopng"]["accs"])
        fp = np.mean(data["fopng"]["accs"])
        print(f"  {label.replace(chr(10),' '):<18} {ef:>8.4f} {fp:>8.4f} {ef-fp:>+8.4f}")
print("\nSaved plots/subrq3_efopng_summary.png")
