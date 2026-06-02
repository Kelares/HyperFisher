"""
dh-sweep_7-12-13.py
────────────────────
dh = 4 / 8 / 16 accuracy + BWT sweep.
Exp 12 (dh=4), Exp 7 (dh=8, from 407.csv), Exp 13 (dh=16).
Justifies the dh=8 choice for Exp 7. Appendix figure.

Output: plots/dh-sweep_7-12-13.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, COLORS, LABELS

RESULTS = "results/"
OUT     = "plots/"


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    def load_pair(fname, dh_filter=None):
        """Load ifopng + adam from a CSV, optionally filtering by hyper_hidden_dim."""
        import ast, pandas as pd
        from collections import defaultdict

        df = pd.read_csv(fname)
        best = defaultdict(dict)
        for _, row in df.iterrows():
            try:
                s = ast.literal_eval(str(row["summary"]))
                c = ast.literal_eval(str(row["config"]))
            except Exception:
                continue
            m = c.get("methods", ["?"])
            if isinstance(m, list): m = m[0]
            if m not in ("ifopng", "adam"): continue
            if dh_filter and c.get("hyper_hidden_dim") != dh_filter: continue
            acc = s.get("best/average_accuracy")
            bwt = s.get("best/bwt")
            tc  = s.get("task_completed", "?"); nt = c.get("num_tasks", "?")
            seed = c.get("seed", "?")
            if acc and tc == nt and acc > 0.05:
                if seed not in best[m] or acc > best[m][seed][0]:
                    best[m][seed] = (acc, bwt)
        result = {}
        for m, seeds in best.items():
            accs = [v[0] for v in seeds.values()]
            bwts = [v[1] for v in seeds.values() if v[1] is not None]
            if len(accs) >= 2:
                result[m] = {"accs": accs, "bwts": bwts}
        return result

    dh4 = load_pair(RESULTS + "412.csv")
    dh8 = load_pair(RESULTS + "407.csv", dh_filter=8)
    dh16 = load_pair(RESULTS + "413.csv")
    dh_all = {4: dh4, 8: dh8, 16: dh16}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(
        "dh Sweep — Split-MNIST SH HyperNetwork (Exps 12, [7], 13)",
        fontsize=11, fontweight="bold",
    )

    x_pos   = {4: 0, 8: 1, 16: 2}
    offsets = {"adam": -0.15, "ifopng": 0.15}

    for ax_i, (metric, ylabel, title_s) in enumerate([
        ("accs", "Avg. Accuracy",   "Average Accuracy"),
        ("bwts", "BWT",             "Backward Transfer"),
    ]):
        ax = axes[ax_i]
        for m in ("adam", "ifopng"):
            xs, means, stds, all_pts = [], [], [], []
            for dh in (4, 8, 16):
                d = dh_all[dh]
                if m not in d or not d[m][metric]:
                    continue
                vals = d[m][metric]
                xs.append(x_pos[dh] + offsets[m])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                all_pts.append((x_pos[dh] + offsets[m], vals))
            if not xs: continue
            col = COLORS[m]
            ax.errorbar(xs, means, yerr=stds, fmt="o-", color=col,
                        capsize=4, linewidth=1.5, markersize=6,
                        label=LABELS[m], zorder=3)
            for xp, pts in all_pts:
                jit = np.linspace(-0.04, 0.04, len(pts))
                for j, p in zip(jit, pts):
                    ax.scatter(xp + j, p, color=col, s=14, alpha=0.5,
                               edgecolors="none", zorder=4)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["dh=4\n(Exp 12)", "dh=8\n(Exp 7)", "dh=16\n(Exp 13)"])
        ax.set_ylabel(ylabel); ax.set_title(title_s, fontsize=10, fontweight="bold")
        ax.axhline(0, color="#333", lw=0.7, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        if metric == "accs": ax.set_ylim(0.45, 1.05)
        else:                ax.set_ylim(-0.55, 0.05)

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "dh-sweep_7-12-13.png")
    plt.close()
    print("Saved dh-sweep_7-12-13.png")

if __name__ == "__main__":
    main()