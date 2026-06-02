"""
generate_plots.py
─────────────────
Generates all thesis experiment plots. Run from the directory containing your CSVs:

    python generate_plots.py

Outputs (saved to ./plots/):
    exp1_exp2_standalone.png
    exp6_cifar100_mh.png
    exp12_13_dh_sweep.png
    stat_cond_normalization.png
    stat_ema_vs_max.png

Requirements: pip install matplotlib numpy pandas scipy
"""

import ast
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── output directory ──────────────────────────────────────────────────────
path = "plots/1,2,4,5,6,12,13"
os.makedirs(path, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

METHOD_COLORS = {
    "ifopng": "#1B6CA8",
    "fopng":  "#5BA3D9",
    "ewc":    "#2E8B57",
    "ogd":    "#E07B2A",
    "ong":    "#C94040",
    "fng":    "#8B5CF6",
    "sgd":    "#888888",
    "adam":   "#444444",
}
METHOD_LABELS = {
    "ifopng": "iFOPNG",
    "fopng":  "FOPNG",
    "ewc":    "EWC",
    "ogd":    "OGD",
    "ong":    "ONG",
    "fng":    "FNG",
    "sgd":    "SGD",
    "adam":   "Adam",
}
SEED_COLORS = {42: "#1B6CA8", 1234: "#E07B2A", 811: "#2E8B57", 2137: "#8B5CF6", 111: "#C94040"}
TASK_BG     = ["#f0f4ff", "#fff8f0", "#f0fff4", "#fff0f8", "#f8f0ff"]


# ── helpers ───────────────────────────────────────────────────────────────

def safe_parse(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return {}


def load_runs(fname, skip_fn=None):
    """
    Returns {method: {seed: [acc, ...]}} then averages per seed.
    skip_fn(method, config_dict, summary_dict) -> bool  (True = skip row)
    """
    df = pd.read_csv(f"results/{fname}")
    raw = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        s = safe_parse(row["summary"])
        c = safe_parse(row["config"])
        m = c.get("methods", ["?"])
        if isinstance(m, list):
            m = m[0]
        seed = c.get("seed", "?")
        acc  = s.get("best/average_accuracy", None)
        bwt  = s.get("best/bwt", None)
        tc   = s.get("task_completed", "?")
        nt   = c.get("num_tasks", "?")
        if skip_fn and skip_fn(m, c, s):
            continue
        if acc is not None and tc == nt and acc > 0.05:
            raw[m][seed].append((acc, bwt))

    result = {}
    for m, seeds in raw.items():
        accs, bwts = [], []
        for seed_runs in seeds.values():
            accs.append(np.mean([r[0] for r in seed_runs]))
            bs = [r[1] for r in seed_runs if r[1] is not None]
            if bs:
                bwts.append(np.mean(bs))
        if len(accs) >= 2:
            result[m] = {
                "acc_mean": np.mean(accs), "acc_std": np.std(accs), "accs": accs,
                "bwt_mean": np.mean(bwts) if bwts else None,
                "bwt_std":  np.std(bwts)  if bwts else None,
                "bwts": bwts,
            }
    return result


def bar_subplot(ax, data, metric, ylabel, title, ylim=None):
    """Bar chart (sorted by mean) with std error bars and seed dots."""
    methods = sorted(data, key=lambda m: -data[m][f"{metric}_mean"])
    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        d    = data[m]
        mean = d[f"{metric}_mean"]
        std  = d[f"{metric}_std"]
        if mean is None:
            continue
        col = METHOD_COLORS.get(m, "#999")
        ax.bar(i, mean, yerr=std, color=col, width=0.6, capsize=3,
               error_kw={"linewidth": 1, "ecolor": "#333", "capthick": 1},
               zorder=3, alpha=0.88)
        pts    = d[f"{metric}s"]
        jitter = np.linspace(-0.12, 0.12, len(pts))
        for j, p in zip(jitter, pts):
            ax.scatter(i + j, p, color="white", s=12, zorder=4,
                       edgecolors="#333", linewidths=0.6)

    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if ylim:
        ax.set_ylim(ylim)


# ─────────────────────────────────────────────────────────────────────────
# # PLOT 1 — Exp 1 + Exp 2: standalone MNIST benchmarks
# # ─────────────────────────────────────────────────────────────────────────
# print("Generating exp1_exp2_standalone.png ...")

# data1 = load_runs("401.csv")
# data2 = load_runs("402.csv")

# fig, axes = plt.subplots(2, 2, figsize=(10, 7))
# fig.suptitle("Standalone Target-Network Benchmarks", fontsize=12,
#              fontweight="bold", y=1.01)

# bar_subplot(axes[0, 0], data1, "acc", "Avg. Accuracy",
#             "Exp 1 — Permuted-MNIST (5T)", ylim=(0.45, 0.95))
# bar_subplot(axes[0, 1], data1, "bwt", "BWT",
#             "Exp 1 — Permuted-MNIST (5T) BWT", ylim=(-0.55, 0.05))
# bar_subplot(axes[1, 0], data2, "acc", "Avg. Accuracy",
#             "Exp 2 — Split-MNIST MH (5T)", ylim=(0.75, 1.02))
# bar_subplot(axes[1, 1], data2, "bwt", "BWT",
#             "Exp 2 — Split-MNIST MH (5T) BWT", ylim=(-0.35, 0.05))

# plt.tight_layout(pad=1.5)
# plt.savefig(f"{path}/exp1_exp2_standalone.png")
# plt.close()


# # ─────────────────────────────────────────────────────────────────────────
# # PLOT 2 — Exp 6: Split-CIFAR100 MH
# # ─────────────────────────────────────────────────────────────────────────
# print("Generating exp6_cifar100_mh.png ...")

# def skip_406(m, c, s):
#     fopt = c.get("first_task_opt", "?")
#     lam  = c.get("lam", "?")
#     # keep only SGD first-task runs; drop EWC with lam=50 (explodes)
#     if fopt == "adam" and m not in ["sgd", "adam"]:
#         return True
#     if m == "ewc" and lam == 50:
#         return True
#     return False

# data6 = load_runs("406.csv", skip_fn=skip_406)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# fig.suptitle(
#     "Exp 6 — Split-CIFAR100 MH Standalone (10T, SGD first-task @ 10⁻²)",
#     fontsize=11, fontweight="bold",
# )
# bar_subplot(axes[0], data6, "acc", "Avg. Accuracy",
#             "Average Accuracy", ylim=(0.1, 0.62))
# bar_subplot(axes[1], data6, "bwt", "BWT",
#             "Backward Transfer (BWT)", ylim=(-0.55, 0.05))

# # annotate EWC lambda note
# if "ewc" in data6:
#     ewc_acc = data6["ewc"]["acc_mean"]
#     axes[0].annotate(
#         "EWC: lam=10\n(lam=50 crashed)",
#         xy=(0, ewc_acc),
#         xytext=(0.5, ewc_acc + 0.06),
#         fontsize=7, color=METHOD_COLORS["ewc"],
#         arrowprops=dict(arrowstyle="->", color=METHOD_COLORS["ewc"], lw=0.8),
#     )

# plt.tight_layout(pad=1.5)
# plt.savefig(f"{path}/exp6_cifar100_mh.png")
# plt.close()


# ─────────────────────────────────────────────────────────────────────────
# PLOT 3 — dₕ sweep: Exps 12 + [7] + 13
# ─────────────────────────────────────────────────────────────────────────
print("Generating exp12_13_dh_sweep.png ...")

def load_dh_data(fname, methods=("adam", "ifopng"), dh_filter=None):
    """
    Load acc + bwt per method from a CSV, deduplicated by seed.
    For seeds with multiple runs, keeps the one with highest accuracy.
    Returns {method: {"accs": [...], "bwts": [...]}}
    """
    df = pd.read_csv(f"results/{fname}")
    # Collect all runs per (method, seed)
    seed_runs = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        s = safe_parse(row["summary"])
        c = safe_parse(row["config"])
        m = c.get("methods", ["?"])
        if isinstance(m, list):
            m = m[0]
        if m not in methods:
            continue
        if dh_filter is not None and c.get("hyper_hidden_dim") != dh_filter:
            continue
        acc = s.get("best/average_accuracy", None)
        bwt = s.get("best/bwt", None)
        tc  = s.get("task_completed", "?")
        nt  = c.get("num_tasks", "?")
        seed = c.get("seed", "?")
        if acc is not None and tc == nt and acc > 0.05:
            seed_runs[m][seed].append((acc, bwt))

    # Deduplicate: one value per seed (best accuracy)
    result = {}
    for m in methods:
        accs, bwts = [], []
        for seed_data in seed_runs[m].values():
            best = max(seed_data, key=lambda x: x[0])
            accs.append(best[0])
            if best[1] is not None:
                bwts.append(best[1])
        if accs:
            result[m] = {"accs": accs, "bwts": bwts}
    return result

METHODS_DH = ("adam", "ifopng")

dh_all = {
    4:  load_dh_data("412.csv",  methods=METHODS_DH),
    8:  load_dh_data("407.csv",  methods=METHODS_DH, dh_filter=8),
    16: load_dh_data("413.csv",  methods=METHODS_DH),
}

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
fig.suptitle(
    "dₕ Sweep — Split-MNIST SH HyperNetwork (Exps 12, [7], 13)",
    fontsize=11, fontweight="bold",
)

x_pos   = {4: 0, 8: 1, 16: 2}
offsets = {"adam": -0.15, "ifopng": 0.15}

for ax_i, (metric, ylabel, title_s) in enumerate([
    ("accs", "Avg. Accuracy",    "Average Accuracy"),
    ("bwts", "BWT",              "Backward Transfer"),
]):
    ax = axes[ax_i]
    for m in METHODS_DH:
        xs, means, stds, all_pts = [], [], [], []
        for dh in [4, 8, 16]:
            d = dh_all[dh]
            if m not in d or not d[m][metric]:
                continue
            vals = d[m][metric]
            xs.append(x_pos[dh] + offsets[m])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            all_pts.append((x_pos[dh] + offsets[m], vals))

        if not xs:
            continue
        col = METHOD_COLORS[m]
        ax.errorbar(xs, means, yerr=stds, fmt="o-", color=col,
                    capsize=4, linewidth=1.5, markersize=6,
                    label=METHOD_LABELS[m], zorder=3)
        for xp, pts in all_pts:
            jit = np.linspace(-0.04, 0.04, len(pts))
            for j, p in zip(jit, pts):
                ax.scatter(xp + j, p, color=col, s=14, alpha=0.5,
                           edgecolors="none", zorder=4)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["dₕ=4\n(Exp 12)", "dₕ=8\n(Exp 7)", "dₕ=16\n(Exp 13)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title_s, fontsize=10, fontweight="bold")
    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    if metric == "accs":
        ax.set_ylim(0.45, 1.05)
    else:
        ax.set_ylim(-0.55, 0.05)

plt.tight_layout(pad=1.5)
plt.savefig(f"{path}/exp12_13_dh_sweep.png")
plt.close()

# # ─────────────────────────────────────────────────────────────────────────
# # PLOT 4 — Conditioning: normalized vs unnormalized (Exp 8 vs Exp 9)
# # ─────────────────────────────────────────────────────────────────────────
# print("Generating stat_cond_normalization.png ...")

# def extract_cond_series(fname, method="ifopng"):
#     """Extract log10_cond_A time series per run from a WandB CSV."""
#     df = pd.read_csv(f"results/{fname}")
#     runs = []
#     for _, row in df.iterrows():
#         s = safe_parse(row["summary"])
#         c = safe_parse(row["config"])
#         m = c.get("methods", ["?"])
#         if isinstance(m, list):
#             m = m[0]
#         if m != method:
#             continue
#         tc = s.get("task_completed", "?")
#         nt = c.get("num_tasks", "?")
#         if tc != nt:
#             continue
#         seed   = c.get("seed")
#         series = {}
#         for k, v in s.items():
#             if "log10_cond_A/" in k and v is not None:
#                 try:
#                     step = int(k.rsplit("/", 1)[-1])
#                     series[step] = v
#                 except ValueError:
#                     continue
#         if series:
#             runs.append({
#                 "seed":    seed,
#                 "series":  series,
#                 "avg_acc": s.get("best/average_accuracy"),
#             })
#     return runs


# runs_8 = extract_cond_series("408.csv")   # normalized
# runs_9 = extract_cond_series("409.csv")   # unnormalized

# # Keep only runs with good accuracy for Exp 8
# runs_8 = [r for r in runs_8 if r["avg_acc"] and r["avg_acc"] > 0.6]

# TASK_STEPS = 40   # 200 logged steps / 5 tasks

# all_vals = [v for r in runs_8 + runs_9 for v in r["series"].values()]
# ymax = max(all_vals) * 1.08 if all_vals else 12

# fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
# fig.suptitle(
#     "Projection Matrix Conditioning: Normalization Ablation (Exp 8 vs Exp 9)",
#     fontsize=11, fontweight="bold",
# )

# for ax, runs, title in [
#     (axes[0], runs_8, "With Normalization (Exp 8)"),
#     (axes[1], runs_9, "Without Normalization (Exp 9)"),
# ]:
#     # task background bands
#     for t in range(5):
#         ax.axvspan(t * TASK_STEPS + 1, (t + 1) * TASK_STEPS,
#                    color=TASK_BG[t], alpha=0.35, zorder=0)
#         ax.axvline((t + 1) * TASK_STEPS, color="#aaa",
#                    linewidth=0.7, linestyle="--", zorder=1)
#         ax.text(t * TASK_STEPS + TASK_STEPS / 2, ymax * 0.96,
#                 f"T{t+1}", ha="center", fontsize=7.5,
#                 color="#555", fontweight="bold")

#     for run in runs:
#         seed  = run["seed"]
#         steps = sorted(run["series"].keys())
#         vals  = [run["series"][s] for s in steps]
#         col   = SEED_COLORS.get(seed, "#888")
#         lbl   = (f"seed={seed}  (acc={run['avg_acc']:.3f})"
#                  if run["avg_acc"] else f"seed={seed}")
#         ax.plot(steps, vals, color=col, linewidth=1.8,
#                 label=lbl, zorder=3, alpha=0.9)
#         peak_i = int(np.argmax(vals))
#         ax.scatter(steps[peak_i], vals[peak_i], color=col,
#                    s=45, zorder=5, marker="D",
#                    edgecolors="white", linewidths=0.8)

#     ax.set_xlabel("Training epoch", fontsize=9)
#     ax.set_ylabel("log₁₀  cond(G⊤F̂G)", fontsize=9)
#     ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
#     ax.set_xlim(1, 200)
#     ax.set_ylim(0, ymax)
#     ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

#     # annotate large explosions in the unnormalized panel
#     if "Without" in title:
#         for run in runs:
#             steps = sorted(run["series"].keys())
#             vals  = [run["series"][s] for s in steps]
#             peak_v = max(vals)
#             if peak_v > 7:
#                 peak_s = steps[int(np.argmax(vals))]
#                 col    = SEED_COLORS.get(run["seed"], "#888")
#                 axes[1].annotate(
#                     f"cond ≈ 10^{peak_v:.1f}",
#                     xy=(peak_s, peak_v),
#                     xytext=(peak_s - 45, peak_v - 1.8),
#                     fontsize=7, color=col,
#                     arrowprops=dict(arrowstyle="->", lw=0.9, color=col),
#                 )

# plt.tight_layout(pad=1.5)
# plt.savefig(f"{path}/stat_cond_normalization.png")
# plt.close()


# # ─────────────────────────────────────────────────────────────────────────
# # PLOT 5 — EMA vs MAX: Wilcoxon (Exps 14 vs 15)
# # ─────────────────────────────────────────────────────────────────────────
# print("Generating stat_ema_vs_max.png ...")

# ema_accs, max_accs = [], []
# ema_seeds, max_seeds = [], []
# for fname, lst, slst in [
#     ("414.csv", ema_accs, ema_seeds),
#     ("415.csv", max_accs, max_seeds),
# ]:
#     df = pd.read_csv(f"results/{fname}")
#     for _, row in df.iterrows():
#         s   = safe_parse(row["summary"])
#         c   = safe_parse(row["config"])
#         acc = s.get("best/average_accuracy")
#         tc  = s.get("task_completed", "?")
#         nt  = c.get("num_tasks", "?")
#         if acc and tc == nt:
#             lst.append(acc)
#             slst.append(c.get("seed"))

# stat_w, p_w = stats.wilcoxon(max_accs, ema_accs, alternative="greater")
# t_stat, p_t = stats.ttest_rel(max_accs, ema_accs, alternative="greater")
# diffs   = [m - e for m, e in zip(max_accs, ema_accs)]
# cohen_d = np.mean(diffs) / np.std(diffs)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
# fig.suptitle(
#     "Exp 14 vs 15 — Fisher Accumulation: EMA vs MAX  (20-task Permuted-MNIST, n=5 seeds)",
#     fontsize=11, fontweight="bold",
# )

# # ── left: paired lines ────────────────────────────────────────────────
# ax = axes[0]
# shared = sorted(set(ema_seeds) & set(max_seeds))
# for seed in shared:
#     ie = ema_seeds.index(seed)
#     im = max_seeds.index(seed)
#     ev = ema_accs[ie]
#     mv = max_accs[im]
#     col = SEED_COLORS.get(seed, "#777")
#     ax.plot([0, 1], [ev, mv], color=col, linewidth=1.5,
#             marker="o", markersize=7, zorder=3, label=f"seed={seed}")
#     ax.annotate(f"  Δ={mv - ev:+.4f}", xy=(1, mv),
#                 fontsize=7, color=col, va="center")

# ax.set_xticks([0, 1])
# ax.set_xticklabels(["EMA\n(iFOPNG_ema)", "MAX\n(iFOPNG)"], fontsize=9)
# ax.set_ylabel("Avg. Accuracy (20 tasks)", fontsize=9)
# ax.set_title("Paired per-seed comparison", fontsize=10, fontweight="bold")
# ax.set_xlim(-0.35, 1.7)
# ylo = min(min(ema_accs), min(max_accs)) - 0.003
# yhi = max(max(ema_accs), max(max_accs)) + 0.008
# ax.set_ylim(ylo, yhi)
# ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

# # ── right: bar + significance bracket ────────────────────────────────
# ax2 = axes[1]
# means = [np.mean(ema_accs), np.mean(max_accs)]
# stds  = [np.std(ema_accs),  np.std(max_accs)]
# ax2.bar([0, 1], means, yerr=stds,
#         color=["#A8C7E8", "#1B6CA8"], width=0.5, capsize=5, zorder=3,
#         alpha=0.88, error_kw={"linewidth": 1.2, "ecolor": "#333", "capthick": 1.2})
# for lst, x in [(ema_accs, 0), (max_accs, 1)]:
#     jit = np.linspace(-0.09, 0.09, len(lst))
#     for j, v in zip(jit, lst):
#         ax2.scatter(x + j, v, color="white", s=20, zorder=4,
#                     edgecolors="#333", linewidths=0.8)

# y_top = max(means) + max(stds) + 0.002
# ax2.plot([0, 0, 1, 1],
#          [y_top, y_top + 0.001, y_top + 0.001, y_top],
#          color="#333", linewidth=1.0)
# star = ("***" if p_t < 0.001 else "**" if p_t < 0.01
#         else "*" if p_t < 0.05 else "ns")
# ax2.text(0.5, y_top + 0.0015, star, ha="center", fontsize=11, fontweight="bold")
# ax2.set_xticks([0, 1])
# ax2.set_xticklabels(["EMA\n(iFOPNG_ema)", "MAX\n(iFOPNG)"], fontsize=9)
# ax2.set_ylabel("Avg. Accuracy (20 tasks)", fontsize=9)
# ax2.set_title("Mean ± std with significance", fontsize=10, fontweight="bold")
# ax2.set_ylim(ylo - 0.001, y_top + 0.006)

# stats_txt = (
#     f"Paired t(4):  t = {t_stat:.2f},  p = {p_t:.5f}\n"
#     f"Wilcoxon:     W = {stat_w:.0f},    p = {p_w:.5f}\n"
#     f"Cohen's d (paired) = {cohen_d:.2f}\n"
#     f"Mean Δ = {np.mean(diffs) * 100:.2f} pp  (MAX > EMA)\n"
#     f"H₀: EMA ≥ MAX  →  REJECTED"
# )
# ax2.text(0.03, 0.05, stats_txt, transform=ax2.transAxes,
#          fontsize=7.5, va="bottom",
#          bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#ccc", alpha=0.95))

# plt.tight_layout(pad=1.5)
# plt.savefig(f"{path}/stat_ema_vs_max.png")
# plt.close()

# # ── summary ───────────────────────────────────────────────────────────────
# print("\nAll plots saved to ./plots/")
# print(f"  exp1_exp2_standalone.png")
# print(f"  exp6_cifar100_mh.png")
# print(f"  exp12_13_dh_sweep.png")
# print(f"  stat_cond_normalization.png")
# print(f"  stat_ema_vs_max.png")
# print(f"\nEMA vs MAX summary:")
# print(f"  EMA: {np.mean(ema_accs):.4f} ± {np.std(ema_accs):.4f}")
# print(f"  MAX: {np.mean(max_accs):.4f} ± {np.std(max_accs):.4f}")
# print(f"  Δ = {np.mean(diffs) * 100:.3f} pp")
# print(f"  Paired t(4) = {t_stat:.3f},  p = {p_t:.5f}")
# print(f"  Wilcoxon W = {stat_w:.0f},  p = {p_w:.5f}")
# print(f"  Cohen's d = {cohen_d:.3f}")