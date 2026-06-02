import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import scipy.stats as stats  # <--- Added for true CI calculation


# ==============================================================================
# W&B SETUP
# ==============================================================================
api     = wandb.Api()
entity  = "michalowski-jb-tilburg-university"
project = "HyperFisher"


# ==============================================================================
# DATA FETCHER
# ==============================================================================

runs = api.runs(
    f"{entity}/{project}",
    filters = {"group": "ogd_lr_11_sweep"},
)
print(len(runs))
LRS = {}
for run in runs:
    lr = run.config.get("lr")
    seed = run.config["seed"]

    if lr not in LRS:
        LRS[lr] = {}

    if run.state == "finished":
        results = run.summary.get("best/results") or run.summary.get("results")
        LRS[lr][seed] = results["acc"]
        print(results)
        break
    else:
        LRS[lr][seed] = "Failed"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("plots", exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
})

LRS = {0.005: {42: 'Failed', 1234: 'Failed', 811: 'Failed'}, 0.001: {42: {'1': [0.639, 0.106, 0.113, 0.1, 0.111, 0.1, 0.133, 0.119, 0.081, 0.07], '10': [0.634, 0.333, 0.374, 0.402, 0.416, 0.396, 0.438, 0.307, 0.333, 0.427], '2': [0.639, 0.329, 0.102, 0.101, 0.113, 0.101, 0.131, 0.135, 0.1, 0.075], '3': [0.64, 0.326, 0.412, 0.102, 0.114, 0.067, 0.127, 0.123, 0.1, 0.064], '4': [0.638, 0.332, 0.397, 0.428, 0.11, 0.061, 0.131, 0.109, 0.098, 0.09], '5': [0.637, 0.324, 0.392, 0.425, 0.461, 0.07, 0.142, 0.11, 0.094, 0.096], '6': [0.639, 0.336, 0.389, 0.422, 0.435, 0.433, 0.131, 0.13, 0.085, 0.09], '7': [0.636, 0.344, 0.385, 0.421, 0.414, 0.409, 0.482, 0.125, 0.092, 0.103], '8': [0.632, 0.342, 0.381, 0.43, 0.424, 0.405, 0.449, 0.397, 0.072, 0.105], '9': [0.637, 0.33, 0.382, 0.419, 0.423, 0.403, 0.442, 0.336, 0.369, 0.103]}, 1234: {'1': [0.609, 0.113, 0.083, 0.105, 0.098, 0.092, 0.135, 0.092, 0.069, 0.089], '10': [0.603, 0.298, 0.297, 0.239, 0.408, 0.319, 0.441, 0.401, 0.379, 0.38], '2': [0.599, 0.303, 0.103, 0.115, 0.083, 0.074, 0.144, 0.101, 0.083, 0.086], '3': [0.6, 0.304, 0.33, 0.111, 0.078, 0.079, 0.145, 0.093, 0.088, 0.086], '4': [0.597, 0.298, 0.316, 0.283, 0.078, 0.078, 0.143, 0.127, 0.074, 0.084], '5': [0.595, 0.297, 0.313, 0.271, 0.443, 0.109, 0.141, 0.124, 0.08, 0.078], '6': [0.598, 0.295, 0.305, 0.25, 0.419, 0.427, 0.141, 0.124, 0.098, 0.076], '7': [0.603, 0.288, 0.312, 0.253, 0.412, 0.359, 0.461, 0.128, 0.101, 0.08], '8': [0.6, 0.295, 0.321, 0.24, 0.42, 0.337, 0.435, 0.438, 0.082, 0.09], '9': [0.598, 0.298, 0.32, 0.248, 0.417, 0.324, 0.445, 0.409, 0.434, 0.087]}, 811: {'1': [0.612, 0.083, 0.084, 0.085, 0.099, 0.081, 0.091, 0.14, 0.09, 0.09], '10': [0.602, 0.407, 0.325, 0.423, 0.364, 0.276, 0.362, 0.293, 0.297, 0.478], '2': [0.619, 0.395, 0.093, 0.087, 0.101, 0.077, 0.087, 0.143, 0.085, 0.091], '3': [0.62, 0.401, 0.444, 0.087, 0.093, 0.077, 0.097, 0.14, 0.085, 0.092], '4': [0.62, 0.396, 0.417, 0.449, 0.089, 0.081, 0.11, 0.146, 0.105, 0.09], '5': [0.613, 0.395, 0.416, 0.441, 0.405, 0.081, 0.103, 0.158, 0.114, 0.098], '6': [0.62, 0.393, 0.413, 0.458, 0.365, 0.431, 0.079, 0.143, 0.105, 0.136], '7': [0.615, 0.384, 0.396, 0.443, 0.358, 0.365, 0.469, 0.146, 0.09, 0.151], '8': [0.611, 0.405, 0.384, 0.433, 0.382, 0.356, 0.377, 0.395, 0.098, 0.148], '9': [0.603, 0.393, 0.321, 0.447, 0.36, 0.346, 0.354, 0.334, 0.38, 0.143]}}, 0.0005: {42: {'1': [0.583, 0.089, 0.097, 0.084, 0.13, 0.107, 0.156, 0.101, 0.092, 0.101], '10': [0.589, 0.295, 0.261, 0.291, 0.34, 0.273, 0.415, 0.276, 0.222, 0.382], '2': [0.584, 0.297, 0.096, 0.101, 0.113, 0.11, 0.153, 0.109, 0.094, 0.098], '3': [0.583, 0.295, 0.271, 0.114, 0.114, 0.127, 0.151, 0.118, 0.086, 0.1], '4': [0.587, 0.296, 0.261, 0.312, 0.106, 0.132, 0.153, 0.124, 0.083, 0.097], '5': [0.589, 0.294, 0.255, 0.294, 0.368, 0.122, 0.146, 0.126, 0.1, 0.092], '6': [0.589, 0.287, 0.266, 0.283, 0.349, 0.321, 0.152, 0.115, 0.099, 0.09], '7': [0.592, 0.283, 0.258, 0.296, 0.352, 0.281, 0.458, 0.113, 0.096, 0.082], '8': [0.592, 0.29, 0.255, 0.283, 0.344, 0.267, 0.422, 0.32, 0.076, 0.091], '9': [0.589, 0.288, 0.256, 0.301, 0.343, 0.268, 0.423, 0.281, 0.248, 0.095]}, 1234: {'1': [0.598, 0.084, 0.115, 0.119, 0.095, 0.075, 0.12, 0.098, 0.076, 0.073], '10': [0.599, 0.244, 0.247, 0.205, 0.228, 0.199, 0.411, 0.365, 0.273, 0.258], '2': [0.597, 0.252, 0.123, 0.095, 0.1, 0.073, 0.123, 0.095, 0.076, 0.072], '3': [0.599, 0.251, 0.263, 0.106, 0.112, 0.071, 0.126, 0.099, 0.069, 0.082], '4': [0.598, 0.249, 0.264, 0.242, 0.116, 0.067, 0.119, 0.106, 0.074, 0.08], '5': [0.598, 0.244, 0.26, 0.229, 0.253, 0.057, 0.12, 0.103, 0.086, 0.084], '6': [0.599, 0.247, 0.251, 0.21, 0.25, 0.254, 0.115, 0.109, 0.089, 0.073], '7': [0.597, 0.247, 0.25, 0.211, 0.238, 0.215, 0.436, 0.114, 0.082, 0.069], '8': [0.598, 0.248, 0.252, 0.205, 0.235, 0.199, 0.425, 0.386, 0.063, 0.076], '9': [0.601, 0.241, 0.247, 0.204, 0.225, 0.202, 0.413, 0.376, 0.301, 0.078]}, 811: {'1': [0.632, 0.068, 0.165, 0.099, 0.086, 0.059, 0.092, 0.169, 0.089, 0.074], '10': [0.628, 0.348, 0.305, 0.254, 0.264, 0.278, 0.326, 0.323, 0.288, 0.433], '2': [0.633, 0.365, 0.18, 0.096, 0.089, 0.068, 0.085, 0.172, 0.103, 0.08], '3': [0.632, 0.353, 0.354, 0.098, 0.081, 0.07, 0.091, 0.187, 0.106, 0.082], '4': [0.631, 0.348, 0.327, 0.276, 0.08, 0.077, 0.09, 0.188, 0.098, 0.064], '5': [0.635, 0.356, 0.326, 0.265, 0.315, 0.08, 0.085, 0.186, 0.082, 0.074], '6': [0.639, 0.356, 0.328, 0.266, 0.284, 0.424, 0.098, 0.191, 0.08, 0.107], '7': [0.631, 0.36, 0.339, 0.263, 0.284, 0.374, 0.401, 0.195, 0.073, 0.133], '8': [0.631, 0.346, 0.328, 0.264, 0.272, 0.338, 0.34, 0.361, 0.089, 0.12], '9': [0.636, 0.346, 0.296, 0.269, 0.266, 0.333, 0.332, 0.332, 0.352, 0.111]}}}

N = 10

def build_R(acc_dict):
    # acc[str(t)][i] = accuracy on task (i+1) AFTER training task t
    # R[task_i, training_step_j] = acc[str(j+1)][task_i]
    R = np.zeros((N, N))
    for t_str, accs in acc_dict.items():
        j = int(t_str) - 1
        for i, v in enumerate(accs):
            R[i, j] = v
    return R

def metrics(acc_dict):
    R      = build_R(acc_dict)
    final  = R[:, -1]              # acc on all tasks after all training
    diag   = np.array([R[i,i] for i in range(N)])
    avg    = np.mean(final)
    bwt    = np.mean([final[i] - diag[i] for i in range(N-1)])
    return avg, bwt, R

# Aggregate
summary = {}
lrs = sorted(LRS.keys())
for lr in lrs:
    accs, bwts, Rs = [], [], []
    for seed, res in LRS[lr].items():
        if res == 'Failed': continue
        a, b, R = metrics(res)
        accs.append(a); bwts.append(b); Rs.append(R)
    summary[lr] = {'accs': accs, 'bwts': bwts, 'Rs': Rs,
                   'n_ok': len(accs), 'n_total': len(LRS[lr])}

best_lr = max((lr for lr in lrs if summary[lr]['n_ok'] > 0),
              key=lambda lr: np.mean(summary[lr]['accs']))

print("Summary:")
for lr in lrs:
    d = summary[lr]
    if d['accs']:
        print(f"  lr={lr}: acc={np.mean(d['accs']):.4f}+-{np.std(d['accs']):.4f}  bwt={np.mean(d['bwts']):.4f}")
    else:
        print(f"  lr={lr}: all failed")

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 12))
fig.suptitle("OGD LR Sweep — Exp 11 (Split-CIFAR100 HyperNetwork, 10 tasks)",
             fontsize=11, fontweight="bold", y=1.01)
gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.35, height_ratios=[0.75, 1.3, 0.85])

ax_grid = fig.add_subplot(gs[0, 0])
ax_acc  = fig.add_subplot(gs[0, 1])
ax_bwt  = fig.add_subplot(gs[0, 2])

# ── status grid ───────────────────────────────────────────────────────────
seeds_all = sorted({s for d in LRS.values() for s in d})
for yi, lr in enumerate(lrs):
    for xi, seed in enumerate(seeds_all):
        res = LRS[lr].get(seed, 'N/A')
        failed = (res == 'Failed')
        fc = '#FCEBEB' if failed else '#EAF3DE'
        ec = '#A32D2D' if failed else '#3B6D11'
        ax_grid.add_patch(plt.Rectangle((xi-0.45, yi-0.45), 0.9, 0.9,
                          fc=fc, ec=ec, lw=1.2, zorder=2))
        if not failed:
            a, _, _ = metrics(res)
            ax_grid.text(xi,  yi+0.1, 'OK',   ha='center', va='center',
                         fontsize=8, color='#3B6D11', fontweight='bold', zorder=3)
            ax_grid.text(xi, yi-0.2, f'{a:.3f}', ha='center', va='center',
                         fontsize=7, color='#3B6D11', zorder=3)
        else:
            ax_grid.text(xi, yi, 'FAIL', ha='center', va='center',
                         fontsize=8, color='#A32D2D', fontweight='bold', zorder=3)

ax_grid.set_xlim(-0.6, len(seeds_all)-0.4); ax_grid.set_ylim(-0.6, len(lrs)-0.4)
ax_grid.set_xticks(range(len(seeds_all))); ax_grid.set_yticks(range(len(lrs)))
ax_grid.set_xticklabels([f's={s}' for s in seeds_all], fontsize=8)
ax_grid.set_yticklabels([str(lr) for lr in lrs], fontsize=8)
ax_grid.set_xlabel("Seed", fontsize=9); ax_grid.set_ylabel("LR", fontsize=9)
ax_grid.set_title("Completion (avg acc)", fontsize=9, fontweight='bold', pad=5)
ax_grid.grid(False)
ax_grid.legend(handles=[mpatches.Patch(fc='#EAF3DE', ec='#3B6D11', label='Done'),
                         mpatches.Patch(fc='#FCEBEB', ec='#A32D2D', label='Failed')],
               fontsize=7.5, loc='lower right')

# ── acc + bwt bars ────────────────────────────────────────────────────────
x = np.arange(len(lrs))
for ax, key, ylabel, ylim, col in [
    (ax_acc, 'accs', 'Avg. Accuracy', (0.0, 0.55), '#1B6CA8'),
    (ax_bwt, 'bwts', 'BWT',           (-0.15, 0.05), '#E07B2A'),
]:
    for xi, lr in enumerate(lrs):
        d = summary[lr]
        if d[key]:
            m, s = np.mean(d[key]), np.std(d[key])
            ax.bar(xi, m, yerr=s, color=col, width=0.55, capsize=4, zorder=3,
                   alpha=0.85, error_kw={'linewidth':1,'ecolor':'#333','capthick':1})
            jit = np.linspace(-0.1, 0.1, len(d[key]))
            for j, v in zip(jit, d[key]):
                ax.scatter(xi+j, v, color='white', s=18, zorder=4,
                           edgecolors='#333', linewidths=0.7)
        else:
            ax.bar(xi, 0, color='#F7C1C1', width=0.55, alpha=0.5, hatch='//', zorder=3)
            ax.text(xi, ylim[0]+0.01, 'all\nfailed', ha='center', fontsize=7,
                    color='#A32D2D', fontweight='bold')
    ax.axhline(0, color='#333', lw=0.7, ls='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([str(lr) for lr in lrs], rotation=25, ha='right')
    ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(ylim)

ax_acc.set_title("Avg. Accuracy vs LR", fontsize=9, fontweight='bold', pad=5)
ax_bwt.set_title("BWT vs LR", fontsize=9, fontweight='bold', pad=5)

# ── heatmaps ─────────────────────────────────────────────────────────────
for i, (lr, ax_h) in enumerate(zip(lrs, [fig.add_subplot(gs[1, i]) for i in range(3)])):
    d = summary[lr]
    if d['Rs']:
        R_mean = np.mean(d['Rs'], axis=0)
        star = ' *' if lr == best_lr else ''
        im = ax_h.imshow(R_mean, vmin=0, vmax=0.65, cmap='RdYlGn',
                         aspect='auto', interpolation='nearest')
        for ti in range(N):   # diagonal box
            ax_h.add_patch(plt.Rectangle((ti-0.5, ti-0.5), 1, 1,
                           fill=False, ec='black', lw=1.5, zorder=2))
        for ti in range(N):
            for tj in range(N):
                v = R_mean[ti, tj]
                tc = 'white' if v < 0.15 or v > 0.52 else 'black'
                ax_h.text(tj, ti, f'{v:.2f}', ha='center', va='center',
                          fontsize=5.5, color=tc)
        plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label='Acc')
        ax_h.set_title(f"lr={lr}{star}  (n={d['n_ok']})", fontsize=8.5,
                       fontweight='bold', pad=4)
    else:
        ax_h.set_facecolor('#FFF0F0')
        ax_h.text(0.5, 0.5, f'lr={lr}\nAll seeds failed', ha='center', va='center',
                  transform=ax_h.transAxes, fontsize=10, color='#A32D2D', fontweight='bold')
        ax_h.set_xticks([]); ax_h.set_yticks([])
    if d['Rs'] or True:
        ax_h.set_xticks(range(N)); ax_h.set_yticks(range(N))
        ax_h.set_xticklabels([f'T{j+1}' for j in range(N)], fontsize=6.5)
        ax_h.set_yticklabels([f'T{i+1}' for i in range(N)], fontsize=6.5)
    if i == 0: ax_h.set_ylabel("Task (evaluated) ->", fontsize=8)
    ax_h.set_xlabel("After training task ->", fontsize=8)

# ── retention profile ─────────────────────────────────────────────────────
ax_ret = fig.add_subplot(gs[2, :])
tasks = np.arange(1, N+1)
cols_lr = {'0.001': '#1B6CA8', str(0.001): '#1B6CA8',
           '0.0005': '#2E8B57', str(0.0005): '#2E8B57'}

for lr in lrs:
    d = summary[lr]
    if not d['Rs']: continue
    col = '#1B6CA8' if lr == 0.001 else '#2E8B57' if lr == 0.0005 else '#888'
    diags  = [[R[i,i]   for i in range(N)] for R in d['Rs']]
    finals = [[R[i,-1]  for i in range(N)] for R in d['Rs']]
    dm = np.mean(diags,  axis=0); ds = np.std(diags,  axis=0)
    fm = np.mean(finals, axis=0); fs = np.std(finals, axis=0)

    ax_ret.plot(tasks, dm, 'o-', color=col, lw=1.6, ms=5, alpha=0.9,
                label=f'lr={lr} just-trained (diag)')
    ax_ret.plot(tasks, fm, 's--', color=col, lw=1.4, ms=4, alpha=0.7,
                label=f'lr={lr} final (after T10)')
    ax_ret.fill_between(tasks, dm-ds, dm+ds, alpha=0.1, color=col)
    ax_ret.fill_between(tasks, fm, dm, alpha=0.06, color='#A32D2D')

ax_ret.axhline(1/10, color='#aaa', lw=0.8, ls=':')
ax_ret.text(10.15, 0.10, 'random', fontsize=7, color='#888', va='center')
ax_ret.set_xticks(tasks); ax_ret.set_xticklabels([f'T{t}' for t in tasks])
ax_ret.set_ylabel("Accuracy"); ax_ret.set_xlabel("Task")
ax_ret.set_ylim(0, 0.75); ax_ret.set_xlim(0.5, N+0.5)
ax_ret.set_title("Retention: just-trained (solid) vs final after all tasks (dashed)",
                 fontsize=9, fontweight='bold', pad=5)
ax_ret.legend(fontsize=7.5, loc='upper right', ncol=2)

plt.savefig("plots/ogd_lr_sweep_exp11.png")
plt.close()
print("Saved plots/ogd_lr_sweep_exp11.png")