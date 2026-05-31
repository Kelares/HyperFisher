import pandas as pd, ast, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

os.makedirs("plots", exist_ok=True)
plt.rcParams.update({
    "font.family":"serif","font.size":9,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.8,"figure.dpi":150,
    "savefig.dpi":200,"savefig.bbox":"tight",
    "axes.grid":True,"grid.alpha":0.2,"grid.linewidth":0.5,
})

METHOD_COLORS = {
    "ifopng":"#1B6CA8","fopng":"#5BA3D9","ewc":"#2E8B57",
    "ogd":"#E07B2A","ong":"#C94040","fng":"#8B5CF6",
    "sgd":"#888888","adam":"#444444",
}
METHOD_LABELS = {
    "ifopng":"iFOPNG","fopng":"FOPNG","ewc":"EWC",
    "ogd":"OGD","ong":"ONG","fng":"FNG","sgd":"SGD","adam":"Adam",
}
ORDER = ["ifopng","fopng","ogd","ong","fng","ewc","adam","sgd"]

def safe_parse(s):
    if pd.isna(s): return {}
    try: return ast.literal_eval(str(s))
    except: return {}

def load_exp(fname, min_acc=0.05, min_seeds=2, skip_fn=None):
    df = pd.read_csv(f"results/{fname}")
    seed_best = defaultdict(lambda: defaultdict(lambda: (-999, None)))
    for _, row in df.iterrows():
        s = safe_parse(row["summary"]); c = safe_parse(row["config"])
        m = c.get("methods",["?"])
        if isinstance(m,list): m=m[0]
        acc = s.get("best/average_accuracy",None)
        bwt = s.get("best/bwt",None)
        tc  = s.get("task_completed","?"); nt = c.get("num_tasks","?")
        seed = c.get("seed","?")
        if skip_fn and skip_fn(m,c,s): continue
        if acc and tc==nt and acc>min_acc:
            if acc > seed_best[m][seed][0]:
                seed_best[m][seed] = (acc, bwt)
    result = {}
    for m, seeds in seed_best.items():
        accs = [v[0] for v in seeds.values()]
        bwts = [v[1] for v in seeds.values() if v[1] is not None]
        if len(accs) >= min_seeds:
            result[m] = {"accs":accs, "bwts":bwts,
                         "acc_mean":np.mean(accs), "acc_std":np.std(accs),
                         "bwt_mean":np.mean(bwts) if bwts else None,
                         "bwt_std":np.std(bwts) if bwts else None}
    return result

def bar_panel(ax, data, metric, ylabel, ylim, title):
    methods = [m for m in ORDER if m in data and data[m][f"{metric}_mean"] is not None]
    methods = sorted(methods, key=lambda m: -(data[m][f"{metric}_mean"] or 0))
    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        d = data[m]; mean = d[f"{metric}_mean"]; std = d[f"{metric}_std"]
        col = METHOD_COLORS.get(m,"#999")
        n = len(d[f"{metric}s"])
        ax.bar(i, mean, yerr=std, color=col, width=0.6, capsize=3,
               error_kw={"linewidth":1,"ecolor":"#333","capthick":1},
               zorder=3, alpha=0.88,
               hatch="/" if n < 3 else "")
        jit = np.linspace(-0.12,0.12,n)
        for j,p in zip(jit, d[f"{metric}s"]):
            ax.scatter(i+j, p, color="white", s=12, zorder=4,
                       edgecolors="#333", linewidths=0.6)
    ax.axhline(0,color="#333",lw=0.7,ls="--",alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m,m) for m in methods],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if ylim: ax.set_ylim(ylim)

# ── Exp 4: Split-CIFAR10 MH standalone ───────────────────────────────────
data4 = load_exp("404.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Exp 4 — Split-CIFAR10 MH Standalone  (5T, Adam first-task @ 10⁻³)",
             fontsize=11, fontweight="bold")
bar_panel(axes[0], data4, "acc", "Avg. Accuracy", (0.55, 0.95),
          "Average Accuracy")
bar_panel(axes[1], data4, "bwt", "BWT", (-0.40, 0.05), "Backward Transfer")
plt.tight_layout(pad=1.5)
plt.savefig("plots/exp4_cifar10_mh_standalone.png")
plt.close()
print("Saved exp4_cifar10_mh_standalone.png")

# ── Exp 7: Split-MNIST SH HN suffocated ──────────────────────────────────
# FNG only has 1 seed — flag it but include
data7 = load_exp("407.csv", min_seeds=1)
# Drop methods with only 1 seed from error bars but keep them
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Exp 7 — Split-MNIST SH HN Suffocated  (dₕ=8, 1406 chunks, 5T×15ep)",
             fontsize=11, fontweight="bold")
bar_panel(axes[0], data7, "acc", "Avg. Accuracy", (0.45, 1.02),
          "Average Accuracy")
bar_panel(axes[1], data7, "bwt", "BWT", (-0.45, 0.05), "Backward Transfer")
# Annotate Adam dominance
ax = axes[0]
adam_mean = data7.get("adam",{}).get("acc_mean", None)
if adam_mean:
    ax.axhline(adam_mean, color=METHOD_COLORS["adam"], lw=1.0,
               ls=":", alpha=0.6)
    ax.text(0.98, adam_mean+0.005, f"Adam {adam_mean:.3f}",
            ha="right", va="bottom", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=METHOD_COLORS["adam"])
axes[0].text(0.02, 0.04,
    "/ = single seed", transform=axes[0].transAxes,
    fontsize=7, color="#666", style="italic")
plt.tight_layout(pad=1.5)
plt.savefig("plots/exp7_mnist_hn_suffocated.png")
plt.close()
print("Saved exp7_mnist_hn_suffocated.png")

# ── Exp 8: Split-CIFAR10 HN standard ─────────────────────────────────────
data8 = load_exp("408.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Exp 8 — Split-CIFAR10 HN Standard  (5T×50ep, AdamW first-task @ 10⁻³)",
             fontsize=11, fontweight="bold")
bar_panel(axes[0], data8, "acc", "Avg. Accuracy", (0.55, 1.0),
          "Average Accuracy")
bar_panel(axes[1], data8, "bwt", "BWT", (-0.12, 0.02), "Backward Transfer")
plt.tight_layout(pad=1.5)
plt.savefig("plots/exp8_cifar10_hn_standard.png")
plt.close()
print("Saved exp8_cifar10_hn_standard.png")

# ── Exp 10: Three-way normalization comparison (Exps 8, 9, 10) ───────────
# Get iFOPNG results from each condition
def get_efopng(fname, min_acc=0.05, min_seeds=1):
    df = pd.read_csv(f"results/{fname}")
    seed_best = {}
    for _, row in df.iterrows():
        s = safe_parse(row["summary"]); c = safe_parse(row["config"])
        m = c.get("methods",["?"])
        if isinstance(m,list): m=m[0]
        if m != "ifopng": continue
        acc = s.get("best/average_accuracy",None)
        bwt = s.get("best/bwt",None)
        tc  = s.get("task_completed","?"); nt = c.get("num_tasks","?")
        seed = c.get("seed","?")
        if acc and tc==nt and acc>min_acc:
            if seed not in seed_best or acc > seed_best[seed][0]:
                seed_best[seed] = (acc, bwt)
    accs = [v[0] for v in seed_best.values()]
    bwts = [v[1] for v in seed_best.values() if v[1] is not None]
    return accs, bwts

accs8, bwts8 = get_efopng("408.csv")   # full norm
accs9, bwts9 = get_efopng("409.csv")   # no norm
accs10, bwts10 = get_efopng("410.csv") # grad-only

conditions = [
    ("No norm\n(Exp 9)",  accs9,  bwts9,  "#C94040"),
    ("Grad-only\n(Exp 10)", accs10, bwts10, "#E07B2A"),
    ("Full norm\n(Exp 8)", accs8,  bwts8,  "#1B6CA8"),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
fig.suptitle("Sub-RQ2: Normalization Ablation — iFOPNG on Split-CIFAR10 HN\n"
             "No normalization vs gradient-only vs full normalization",
             fontsize=10, fontweight="bold")

for ax_i, (ylabel, metric_idx, ylim, title) in enumerate([
    ("Avg. Accuracy", 0, (0.35, 1.0), "Average Accuracy"),
    ("BWT",           1, (-0.25, 0.05), "Backward Transfer"),
]):
    ax = axes[ax_i]
    for xi, (label, accs, bwts, col) in enumerate(conditions):
        vals = accs if metric_idx==0 else bwts
        if not vals: continue
        mean, std = np.mean(vals), np.std(vals)
        ax.bar(xi, mean, yerr=std, color=col, width=0.55, capsize=4,
               zorder=3, alpha=0.88,
               error_kw={"linewidth":1.2,"ecolor":"#333","capthick":1.2})
        jit = np.linspace(-0.1, 0.1, len(vals))
        for j,v in zip(jit, vals):
            ax.scatter(xi+j, v, color="white", s=16, zorder=4,
                       edgecolors="#333", linewidths=0.7)
        # mean label on bar
        ax.text(xi, mean + (std or 0) + 0.01, f"{mean:.3f}",
                ha="center", fontsize=8, fontweight="bold", color=col)
    ax.axhline(0,color="#333",lw=0.7,ls="--",alpha=0.5)
    if metric_idx==0:
        ax.axhline(0.50, color="#888", lw=0.8, ls=":", alpha=0.7)
        ax.text(2.45, 0.51, "random", ha="right", fontsize=7, color="#888")
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in conditions])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylim(ylim)

plt.tight_layout(pad=1.5)
plt.savefig("plots/exp_norm_ablation.png")
plt.close()
print("Saved exp_norm_ablation.png")

# Summary numbers
print("\nNorm ablation summary (iFOPNG):")
for label, accs, bwts, _ in conditions:
    if accs:
        print(f"  {label.replace(chr(10),' '):<20}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  bwt={np.mean(bwts):.4f} n={len(accs)}")