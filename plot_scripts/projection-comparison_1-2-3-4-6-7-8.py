"""
projection-comparison_1-2-3-4-6-7-8.py
──────────────────────────────────────
iFOPNG vs FOPNG cross-benchmark summary. Sub-RQ3 main figure.
Seven benchmarks: five standalone + two HN.
Grouped bars per benchmark with paired t-test significance annotations
and per-benchmark seed counts.

Statistical note
  Equal n  → paired t-test on per-seed differences (same seeds assumed).
  Unequal n → Welch's independent t-test.
  All tests two-sided. Cohen's d: paired = mean(Δ)/std(Δ);
  independent = (μ₁−μ₂)/pooled σ.
  df=2 (n=3) tests are low-powered; treat "ns" with caution there.

Output: plots/projection-comparison_1-2-3-4-6-7-8.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from utils import STYLE, load_exp, skip_exp6, skip_exp7, COLORS

RESULTS = "results/"
OUT     = "plots/"

# ── data loading ──────────────────────────────────────────────────────────────

def _load_pair(fname, skip_fn=None):
    data = load_exp(fname, min_seeds=2, skip_fn=skip_fn)
    return {m: data[m] for m in ("ifopng", "fopng") if m in data}


# ── statistics ────────────────────────────────────────────────────────────────

def compute_stats(accs_if, accs_fp):
    a = np.array(accs_if, dtype=float)
    b = np.array(accs_fp, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    if n1 == n2:
        t, p = stats.ttest_rel(a, b)
        diffs = a - b
        sd    = diffs.std(ddof=1)
        d     = float(diffs.mean() / sd) if sd > 0 else 0.0
        test_type = "paired"
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt(((n1-1)*a.std(ddof=1)**2 +
                          (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
        d = float((a.mean()-b.mean()) / pooled) if pooled > 0 else 0.0
        test_type = "independent"
    return dict(t=float(t), p=float(p), d=d,
                n_if=n1, n_fp=n2, test_type=test_type)


def sig_stars(p, n_min):
    if   p < 0.001: label = "***"
    elif p < 0.01:  label = "**"
    elif p < 0.05:  label = "*"
    else:           label = "ns"
    if n_min < 4:
        label += "†"
    return label


# ── plot helpers ──────────────────────────────────────────────────────────────

def annotate_sig(ax, x_left, x_right, y_base, result):
    if result is None:
        return
    stars = sig_stars(result["p"], min(result["n_if"], result["n_fp"]))
    tick, gap = 0.018, 0.008
    y0, y1 = y_base, y_base + tick
    ax.plot([x_left, x_left, x_right, x_right],
            [y0,     y1,     y1,      y0],
            lw=0.8, color="#333", clip_on=False)
    color  = "#111" if stars not in ("ns", "ns†") else "#999"
    weight = "bold" if stars not in ("ns", "ns†") else "normal"
    ax.text((x_left+x_right)/2, y1+gap, stars,
            ha="center", va="bottom", fontsize=8.5,
            color=color, fontweight=weight, clip_on=False)


def _n_label(data):
    n_if = len(data["ifopng"]["accs"]) if "ifopng" in data else 0
    n_fp = len(data["fopng"]["accs"])  if "fopng"  in data else 0
    if n_if == n_fp and n_if > 0:
        return f"n = {n_if}"
    parts = []
    if n_if: parts.append(f"iF={n_if}")
    if n_fp: parts.append(f"FP={n_fp}")
    return "n: " + "/".join(parts) if parts else ""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    benchmarks = [
        ("Perm-MNIST",    _load_pair(RESULTS + "401.csv"),                    "standalone"),
        ("MNIST MH",      _load_pair(RESULTS + "402.csv"),                    "standalone"),
        ("MNIST SH",      _load_pair(RESULTS + "403.csv"),                    "standalone"),  # ← NEW
        ("CIFAR-10 MH",   _load_pair(RESULTS + "404.csv"),                    "standalone"),
        ("CIFAR-100 MH",  _load_pair(RESULTS + "406.csv", skip_fn=skip_exp6), "standalone"),
        ("MNIST HN",      _load_pair(RESULTS + "407.csv", skip_fn=skip_exp7), "HN"),
        ("CIFAR-10 HN",   _load_pair(RESULTS + "408.csv"),                    "HN"),
    ]

    COL_IF = COLORS["ifopng"]
    COL_FP = COLORS["fopng"]
    WIDTH  = 0.32
    x      = np.arange(len(benchmarks))   # 0..6

    # ── stats ─────────────────────────────────────────────────────────────
    stat_results = []
    for _, data, _ in benchmarks:
        if "ifopng" in data and "fopng" in data:
            stat_results.append(
                compute_stats(data["ifopng"]["accs"], data["fopng"]["accs"])
            )
        else:
            stat_results.append(None)

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2))   # wider for 7 benchmarks
    fig.suptitle(
        "Sub-RQ3: iFOPNG vs FOPNG — Does parameter inertia improve retention?\n"
        "Standalone benchmarks (left of divider) vs HN benchmarks (right)",
        fontsize=10, fontweight="bold",
    )

    bracket_tops = []

    for xi, (label, data, btype) in enumerate(benchmarks):
        group_top = 0.0
        for offset, key, col in [(-WIDTH/2, "ifopng", COL_IF),
                                  (+WIDTH/2, "fopng",  COL_FP)]:
            if key not in data:
                continue
            mean = data[key]["acc_mean"]
            std  = data[key]["acc_std"]
            ax.bar(xi + offset, mean, width=WIDTH, color=col,
                   yerr=std, capsize=3, zorder=3, alpha=0.88,
                   error_kw={"linewidth": 0.8, "ecolor": "#333", "capthick": 0.8})
            pts = data[key]["accs"]
            jit = np.linspace(-0.05, 0.05, len(pts))
            for j, v in zip(jit, pts):
                ax.scatter(xi + offset + j, v, color="white", s=10,
                           zorder=4, edgecolors="#333", linewidths=0.5)
            group_top = max(group_top, mean + std)
        bracket_tops.append(group_top)

    # ── significance brackets ─────────────────────────────────────────────
    for xi, result in enumerate(stat_results):
        if result is None:
            continue
        annotate_sig(ax,
                     xi - WIDTH/2,
                     xi + WIDTH/2,
                     bracket_tops[xi] + 0.015,
                     result)

    # ── divider — 5 standalone | 2 HN ────────────────────────────────────
    ax.axvline(4.5, color="#999", lw=1.0, ls="--", alpha=0.6)
    ax.text(4.6, 1.0, "HN →", fontsize=7.5, color="#666", va="top",
            transform=ax.get_xaxis_transform())

    # ── x-tick labels ─────────────────────────────────────────────────────
    tick_labels = [f"{label}\n{_n_label(data)}"
                   for label, data, _ in benchmarks]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel("Avg. Accuracy", fontsize=10)

    y_ceil = max(bracket_tops) + 0.015 + 0.018 + 0.008 + 0.04
    ax.set_ylim(0.0, max(y_ceil, 1.08))
    ax.set_title("iFOPNG vs FOPNG per benchmark", fontsize=10,
                 fontweight="bold", pad=6)

    ax.legend(handles=[
        mpatches.Patch(color=COL_IF,
                       label=r"iFOPNG  ($F_c = \hat{F}_\mathrm{new} + \hat{F}_\mathrm{old}$)"),
        mpatches.Patch(color=COL_FP,
                       label=r"FOPNG   ($\hat{F}_\mathrm{new}$ only)"),
    ], fontsize=8.5, loc="lower left")

    ax.text(0.99, 0.01,
            "*** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05  "
            "† df=2 (n=3), low power\n"
            "Paired t-test (equal n); Welch independent (unequal n); two-sided",
            transform=ax.transAxes, fontsize=6.5,
            ha="right", va="bottom", color="#666")

    plt.tight_layout(pad=1.5)
    out_path = OUT + "projection-comparison_1-2-3-4-6-7-8.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # ── console summary ───────────────────────────────────────────────────
    col = (f"{'Benchmark':<14} {'iFOPNG':>7} {'FOPNG':>7} {'Δ':>7}  "
           f"{'t':>6}  {'p':>7}  {'d':>6}  {'sig':>5}  {'test':>11}  n")
    print(f"\n{col}")
    print("─" * len(col))
    for (label, data, _), result in zip(benchmarks, stat_results):
        if "ifopng" not in data or "fopng" not in data:
            continue
        m_if  = np.mean(data["ifopng"]["accs"]) * 100
        m_fp  = np.mean(data["fopng"]["accs"])  * 100
        delta = m_if - m_fp
        name  = label.replace("\n", " ")
        if result:
            stars = sig_stars(result["p"], min(result["n_if"], result["n_fp"]))
            n_str = (f"{result['n_if']}" if result["n_if"] == result["n_fp"]
                     else f"{result['n_if']}/{result['n_fp']}")
            print(f"{name:<14} {m_if:>6.1f}%  {m_fp:>6.1f}%  {delta:>+6.1f}pp  "
                  f"{result['t']:>6.2f}  {result['p']:>7.4f}  {result['d']:>6.2f}  "
                  f"{stars:>5}  {result['test_type']:>11}  n={n_str}")
        else:
            print(f"{name:<14} {m_if:>6.1f}%  {m_fp:>6.1f}%  {delta:>+6.1f}pp  "
                  f"{'—':>6}  {'—':>7}  {'—':>6}  {'—':>5}  {'—':>11}  "
                  "(insufficient n)")


if __name__ == "__main__":
    main()