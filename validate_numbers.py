"""
validate_thesis_numbers.py
──────────────────────────
Reproduces every numeric claim in the thesis from the CSV exports.

PASS  — matches within 0.15pp (display rounding)
WARN  — matches within 0.5pp  (seed-subset or run-selection delta)
FAIL  — material discrepancy — investigate before submission
INFO  — diagnostic, no pass/fail verdict
STAT  — paired t-test or effect size check

Usage:
    python validate_thesis_numbers.py
"""

import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR  = Path("~/Desktop/HyperFisher/results/")
TOL_EXACT = 0.0015   # 0.15pp → PASS
TOL_WARN  = 0.005    # 0.5pp  → WARN

PASS_COUNT = WARN_COUNT = FAIL_COUNT = 0

# ── Core helpers ──────────────────────────────────────────────────────────────

def safe_parse(s):
    if pd.isna(s): return {}
    try: return ast.literal_eval(str(s))
    except: return {}


def load_exp(fname, skip_fn=None, min_acc=0.05,
             req_normalize=None, req_chunk=None, req_dh=None,
             req_first_opt=None, methods_filter=None):
    """Return (accs, bwts) where accs = {method: {seed: best_acc}}."""
    df = pd.read_csv(DATA_DIR / fname)
    seed_best = defaultdict(lambda: defaultdict(lambda: -999.0))
    seed_bwt  = defaultdict(lambda: defaultdict(lambda: None))

    for _, row in df.iterrows():
        s = safe_parse(row["summary"]); c = safe_parse(row["config"])
        m = c.get("methods", ["?"])
        if isinstance(m, list): m = m[0]
        if methods_filter and m not in methods_filter: continue
        seed = c.get("seed", "?")
        acc  = s.get("best/average_accuracy", None)
        bwt  = s.get("best/bwt", None)
        tc   = s.get("task_completed", "?")
        nt   = c.get("num_tasks", "?")
        if skip_fn and skip_fn(m, c, s): continue
        if acc is None or tc != nt or acc <= min_acc: continue
        if req_normalize is not None and c.get("normalize") != req_normalize: continue
        if req_chunk     is not None and c.get("chunk_size") != req_chunk:     continue
        if req_dh        is not None and c.get("hyper_hidden_dim") != req_dh:  continue
        if req_first_opt is not None and c.get("first_task_opt") != req_first_opt: continue
        if acc > seed_best[m][seed]:
            seed_best[m][seed] = acc
            seed_bwt[m][seed]  = bwt

    accs, bwts = {}, {}
    for method, seeds in seed_best.items():
        valid = {s: v for s, v in seeds.items() if v > -999}
        if valid:
            accs[method] = valid
            bwts[method] = {s: seed_bwt[method][s] for s in valid
                            if seed_bwt[method][s] is not None}
    return accs, bwts


def mean_of(d, m):
    if m not in d or not d[m]: return None
    return np.mean(list(d[m].values()))

def n_of(d, m):
    return len(d.get(m, {}))


def check(label, computed, thesis, tol_e=TOL_EXACT, tol_w=TOL_WARN,
          computed_n=None, thesis_n=None):
    global PASS_COUNT, WARN_COUNT, FAIL_COUNT
    if computed is None:
        print(f"  SKIP  {label}  [no data]"); return
    diff = abs(computed - thesis)
    n_str = f"  n={computed_n}" if computed_n is not None else ""
    if thesis_n and computed_n and computed_n != thesis_n:
        n_str += f" (thesis n={thesis_n})"
    if   diff <= tol_e: tag = "PASS "; PASS_COUNT += 1
    elif diff <= tol_w: tag = "WARN "; WARN_COUNT += 1
    else:               tag = "FAIL "; FAIL_COUNT += 1
    print(f"  {tag} {label:<55} "
          f"computed={computed*100:6.2f}%  thesis={thesis*100:6.2f}%  "
          f"Δ={diff*100:+.2f}pp{n_str}")


def check_bwt(label, bwt_dict, method, thesis_bwt):
    vals = list(bwt_dict.get(method, {}).values())
    if not vals:
        print(f"  SKIP  {label}  [no BWT data]"); return
    check(label, np.mean(vals) + 1, thesis_bwt + 1, tol_e=0.002, tol_w=0.008)


def check_stat(label, t_c, p_c, dz_c, t_th=None, p_th=None, dz_th=None):
    global PASS_COUNT, WARN_COUNT, FAIL_COUNT
    parts = []
    if t_th is not None and t_c is not None:
        d = abs(t_c - t_th); tag = "PASS" if d < 0.1 else "WARN" if d < 0.5 else "FAIL"
        if tag=="PASS": PASS_COUNT+=1
        elif tag=="WARN": WARN_COUNT+=1
        else: FAIL_COUNT+=1
        parts.append(f"t={t_c:.2f}({tag},thesis={t_th:.2f})")
    if p_th is not None and p_c is not None:
        ok = (p_c < 0.05) == (p_th < 0.05)
        tag = "PASS" if ok else "FAIL"
        if tag=="PASS": PASS_COUNT+=1
        else: FAIL_COUNT+=1
        parts.append(f"p={p_c:.3f}({tag},thesis={p_th:.3f})")
    if dz_th is not None and dz_c is not None:
        d = abs(dz_c - dz_th); tag = "PASS" if d < 0.15 else "WARN" if d < 0.5 else "FAIL"
        if tag=="PASS": PASS_COUNT+=1
        elif tag=="WARN": WARN_COUNT+=1
        else: FAIL_COUNT+=1
        parts.append(f"dz={dz_c:.2f}({tag},thesis={dz_th:.2f})")
    print(f"  STAT  {label:<55} {', '.join(parts)}")


def paired_t(a, b, seeds=None):
    seeds = seeds or sorted(set(a) & set(b))
    if len(seeds) < 2: return None, None, None
    av = np.array([a[s] for s in seeds])
    bv = np.array([b[s] for s in seeds])
    d  = av - bv
    t, p = stats.ttest_rel(av, bv)
    return t, p, d.mean()/d.std(ddof=1)


def section(t):    print(f"\n{'═'*70}\n  {t}\n{'═'*70}")
def subsection(t): print(f"\n  ── {t} ──")


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TARGET-NETWORK  (Exps 401–406)
# Numbers sourced from: standalone_targetnetwork.tex figure captions
#                       + full_benchmark.tex table
# ═══════════════════════════════════════════════════════════════════════════════

section("STANDALONE: Permuted-MNIST 5T  (Exp 401)")
# Source: standalone_targetnetwork.tex B.1 caption + full_benchmark.tex
accs401, bwts401 = load_exp("401.csv")
for m, tv in [("ifopng",0.861),("fopng",0.858),("ewc",0.837),("sgd",0.849),
              ("ogd",0.806),("adam",0.677),("fng",0.685),("ong",0.624)]:
    check(f"Exp401 {m.upper()}", mean_of(accs401,m), tv, computed_n=n_of(accs401,m))

section("STANDALONE: Split-MNIST MH 5T  (Exp 402)")
# Source: standalone_targetnetwork.tex B.1 caption + full_benchmark.tex
accs402, bwts402 = load_exp("402.csv")
for m, tv in [("ifopng",0.959),("fopng",0.953),("ewc",0.933),("ogd",0.965),
              ("ong",0.868),("sgd",0.959),("fng",0.846),("adam",0.953)]:
    check(f"Exp402 {m.upper()}", mean_of(accs402,m), tv, computed_n=n_of(accs402,m))

subsection("E3/E18: FOPNG MH seed count — thesis reports 95.3% from 3 common seeds")
fopng402_3 = np.mean([accs402["fopng"][s] for s in [42,811,1234]
                       if s in accs402.get("fopng",{})]) if "fopng" in accs402 else None
fopng402_5 = mean_of(accs402,"fopng")
print(f"  INFO  FOPNG MH: 5-seed mean={fopng402_5*100:.2f}%  "
      f"3-common-seed mean={fopng402_3*100:.2f}%  thesis=95.3%")

section("STANDALONE: Split-MNIST SH 5T  (Exp 403)")
# Source: standalone_targetnetwork.tex B.4 caption (the authoritative SH numbers)
accs403, bwts403 = load_exp("403.csv")
for m, tv in [("ifopng",0.800),("fopng",0.787),("ewc",0.761),("ogd",0.667),
              ("sgd",0.638),("adam",0.627),("fng",0.618),("ong",0.605)]:
    check(f"Exp403 {m.upper()}", mean_of(accs403,m), tv, computed_n=n_of(accs403,m))

subsection("Cross-check: results.tex Sub-RQ3 SH claims (may differ from appendix)")
# results.tex L252 says iFOPNG=80.2%, FOPNG=77.1% — appendix says 80.0% / 78.7%
for m, tv_results, tv_appendix in [("ifopng",0.802,0.800),("fopng",0.771,0.787)]:
    c = mean_of(accs403,m)
    if c:
        diff_r = abs(c-tv_results)*100
        diff_a = abs(c-tv_appendix)*100
        closer = "results.tex" if diff_r < diff_a else "appendix"
        print(f"  INFO  Exp403 {m.upper()}: computed={c*100:.2f}%  "
              f"results.tex={tv_results*100:.1f}% (Δ{diff_r:.2f}pp)  "
              f"appendix={tv_appendix*100:.1f}% (Δ{diff_a:.2f}pp)  → closer to {closer}")

section("STANDALONE: Split-CIFAR10 MH 5T  (Exp 404)")
# Source: standalone_targetnetwork.tex B.2 caption
# Adam first_task, TargetNetwork, normalize=False
accs404, bwts404 = load_exp("404.csv")
for m, tv in [("ifopng",0.812),("ewc",0.819),("fopng",0.674),("fng",0.698),
              ("ogd",0.713),("ong",0.686),("sgd",0.690),("adam",0.681)]:
    check(f"Exp404 {m.upper()}", mean_of(accs404,m), tv, computed_n=n_of(accs404,m))

subsection("E1/E2: iFOPNG and gap values — appendix says 81.1% / 13.7pp; table/text say 81.2% / 13.8pp")
ci = mean_of(accs404,"ifopng"); cf = mean_of(accs404,"fopng")
if ci and cf:
    print(f"  INFO  iFOPNG={ci*100:.2f}%  FOPNG={cf*100:.2f}%  gap={abs(ci-cf)*100:.2f}pp")
    check("E1 iFOPNG vs table claim (81.2%)", ci, 0.812)
    check("E1 iFOPNG vs caption claim (81.1%)", ci, 0.811)

section("STANDALONE: Split-CIFAR100 MH 10T  (Exp 406) FIX THE WANDB GARBAGE LATER.")
# Source: standalone_targetnetwork.tex B.3 caption + full_benchmark.tex
def skip_406(m, c, s):
    return (c.get("first_task_opt")=="adam" and m not in ["sgd","adam"]) or \
           (m=="ewc" and c.get("lam")==50)
accs406, bwts406 = load_exp("406.csv", skip_fn=skip_406)
for m, tv in [("ifopng",0.427),("fopng",0.371),("ewc",0.422),
              ("ogd",0.381),("ong",0.323),("sgd",0.394),("adam",0.449)]:
    check(f"Exp406 {m.upper()}", mean_of(accs406,m), tv, computed_n=n_of(accs406,m))

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-RQ1: CIFAR-10 HyperNetwork  (Exp 408)
# Numbers from: results.tex L27–L50, full_benchmark.tex table
# ═══════════════════════════════════════════════════════════════════════════════

section("SUB-RQ1: CIFAR-10 HyperNetwork  (Exp 408)")
accs408, bwts408 = load_exp(
    "408.csv", req_normalize=True, req_chunk=6000, req_dh=32, req_first_opt="adamw"
)
# Thesis values from results.tex and full_benchmark.tex
for m, tv in [("adam",0.901),("ifopng",0.856),("sgd",0.841),("ewc",0.815),
              ("fng",0.809),("ogd",0.783),("fopng",0.775),("ong",0.672)]:
    check(f"Exp408 {m.upper()}", mean_of(accs408,m), tv, computed_n=n_of(accs408,m))

check_bwt("Exp408 Adam BWT (thesis 0.000)",   bwts408, "adam",   0.000)
check_bwt("Exp408 iFOPNG BWT (thesis -0.013)", bwts408, "ifopng",-0.013)

subsection("Degenerate seeds (EWC seed=1234, OGD seed=2137) — disclosure check")
for m, seed in [("ewc",1234),("ogd",2137)]:
    if m in accs408 and seed in accs408[m]:
        v = accs408[m][seed]
        print(f"  INFO  Exp408 {m.upper()} seed={seed}: acc={v:.4f} "
              f"{'← DEGENERATE (≈random) — included in mean without disclosure' if v < 0.52 else '← OK'}")

subsection("E22: iFOPNG value — results.tex L34=85.6% vs L195=85.8%")
v = mean_of(accs408,"ifopng")
print(f"  INFO  Exp408 iFOPNG computed={v*100:.2f}% | L34=85.6% | L195=85.8% "
      f"| closer to {'L34' if abs(v-0.856)<abs(v-0.858) else 'L195'}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-RQ2: Normalization ablation  (Exps 409, 410 + 408 full-norm)
# Numbers from: results.tex L180–200, hyper_parameters.tex G.3 table
# ═══════════════════════════════════════════════════════════════════════════════

section("SUB-RQ2: Normalization Ablation  (Exps 409, 410, 408)")
accs409, bwts409 = load_exp("409.csv", req_normalize=False,  req_chunk=6000, req_dh=32,
                             methods_filter=["ifopng","fopng"])
accs410, bwts410 = load_exp("410.csv", req_normalize=True,   req_chunk=6000, req_dh=32,
                             methods_filter=["ifopng"])

check("Cond1 no-norm   iFOPNG acc (thesis 72.2%)", mean_of(accs409,"ifopng"), 0.722,
      computed_n=n_of(accs409,"ifopng"), thesis_n=3)
check("Cond2 grad-only iFOPNG acc (thesis 68.9%)", mean_of(accs410,"ifopng"), 0.689,
      computed_n=n_of(accs410,"ifopng"), thesis_n=3)
check("Cond3 full-norm iFOPNG acc (thesis 85.6%, L34)", mean_of(accs408,"ifopng"), 0.856,
      computed_n=n_of(accs408,"ifopng"), thesis_n=5)
check("Cond3 full-norm iFOPNG acc (thesis 85.8%, L195)", mean_of(accs408,"ifopng"), 0.858,
      computed_n=n_of(accs408,"ifopng"), thesis_n=5)
check_bwt("Cond1 no-norm   BWT (thesis -0.070)", bwts409, "ifopng", -0.070)
check_bwt("Cond2 grad-only BWT (thesis -0.060)", bwts410, "ifopng", -0.060)
check_bwt("Cond3 full-norm BWT (thesis -0.013)", bwts408, "ifopng", -0.013)

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-RQ3: iFOPNG vs FOPNG paired t-tests  (Exps 401–408)
# Numbers from: results.tex L238–280
# ═══════════════════════════════════════════════════════════════════════════════

section("SUB-RQ3: iFOPNG vs FOPNG — paired t-tests")

subsection("Permuted-MNIST 5T  (Exp 401, n=3 each)")
check("Exp401 iFOPNG acc (thesis 86.1%)", mean_of(accs401,"ifopng"), 0.861,
      computed_n=n_of(accs401,"ifopng"), thesis_n=3)
check("Exp401 FOPNG  acc (thesis 85.8%)", mean_of(accs401,"fopng"),  0.858,
      computed_n=n_of(accs401,"fopng"),  thesis_n=3)

subsection("FIX THOSE VALUES!!!: Split-MNIST SH 5T  (Exp 403) — paired t-test (thesis: t(4)=24.07, p<.001, dz=10.76)")
t,p,dz = paired_t(accs403.get("ifopng",{}), accs403.get("fopng",{}))
if t: check_stat("Exp403 SH", t,p,dz, t_th=24.07, p_th=0.001, dz_th=10.76)
check("Exp403 iFOPNG acc (thesis 80.2%, results.tex)", mean_of(accs403,"ifopng"), 0.802,
      computed_n=n_of(accs403,"ifopng"))
check("Exp403 FOPNG  acc (thesis 77.1%, results.tex)", mean_of(accs403,"fopng"),  0.771,
      computed_n=n_of(accs403,"fopng"))

subsection("Split-CIFAR10 MH 5T  (Exp 404) — paired t-test (thesis: t(4)=3.54, p=.024, dz=1.58)")
t,p,dz = paired_t(accs404.get("ifopng",{}), accs404.get("fopng",{}))
if t: check_stat("Exp404 CIFAR10", t,p,dz, t_th=3.54, p_th=0.024, dz_th=1.58)

subsection("Split-CIFAR100 10T  (Exp 406, n=3 common) — paired t-test (thesis: t(2)=6.08, p=.026)")
common406 = sorted(set(accs406.get("ifopng",{})) & set(accs406.get("fopng",{})))
t,p,dz = paired_t(accs406.get("ifopng",{}), accs406.get("fopng",{}), seeds=common406)
if t: check_stat("Exp406 CIFAR100", t,p,dz, t_th=6.08, p_th=0.026)
check("Exp406 iFOPNG acc (thesis 42.7%)", mean_of(accs406,"ifopng"), 0.427, computed_n=n_of(accs406,"ifopng"))
check("Exp406 FOPNG  acc (thesis 37.1%)", mean_of(accs406,"fopng"),  0.371, computed_n=n_of(accs406,"fopng"))

subsection("MNIST HN  (Exp 407, excl seed 2137) — thesis: t(4)=11.80, p<.001, dz=5.28")
accs407, bwts407 = load_exp("407.csv", req_normalize=True, req_chunk=64, req_dh=8)
accs407x = {m: {s:v for s,v in sd.items() if s!=2137} for m,sd in accs407.items()}
t,p,dz = paired_t(accs407x.get("ifopng",{}), accs407x.get("fopng",{}))
if t: check_stat("Exp407 MNIST HN", t,p,dz, t_th=11.80, p_th=0.001, dz_th=5.28)

subsection("CIFAR-10 HN  (Exp 408) — no stat test stated in thesis")
check("Exp408 iFOPNG acc (thesis 85.6%)", mean_of(accs408,"ifopng"), 0.856, computed_n=n_of(accs408,"ifopng"))
check("Exp408 FOPNG  acc (thesis 77.5%)", mean_of(accs408,"fopng"),  0.775, computed_n=n_of(accs408,"fopng"))

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-RQ4: EMA vs MAX  (Exps 414, 415)
# Numbers from: results.tex L290–320
# ═══════════════════════════════════════════════════════════════════════════════

section("SUB-RQ4: EMA vs MAX  (Exps 414, 415)")
accs414, _ = load_exp("414.csv", methods_filter=["ifopng_ema"])
accs415, _ = load_exp("415.csv", methods_filter=["ifopng"])

ema = accs414.get("ifopng_ema",{})
mx  = accs415.get("ifopng",{})
check("EMA  avg-acc (thesis 57.1%)", np.mean(list(ema.values())) if ema else None, 0.571,
      computed_n=len(ema), thesis_n=5)
check("MAX  avg-acc (thesis 58.7%)", np.mean(list(mx.values()))  if mx  else None, 0.587,
      computed_n=len(mx),  thesis_n=5)

t,p,dz = paired_t(mx, ema)
if t: check_stat("EMA vs MAX paired t (thesis: t(4)=14.43, p<.001, dz=6.45)",
                 t,p,dz, t_th=14.43, p_th=0.001, dz_th=6.45)
else: print(f"  SKIP  EMA vs MAX t-test — seeds: EMA={sorted(ema)}, MAX={sorted(mx)}")

subsection("E4: appendix reports 'Cohen d=7.22' — should be d_z=6.45")
if dz: print(f"  INFO  d_z={dz:.2f}  (results.tex=6.45 ✓;  appendix wrongly labels it 'Cohen d=7.22')")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION ABLATION  (Exps 421, 422, 423)
# Numbers from: results.tex, additional_experiments.tex
# ═══════════════════════════════════════════════════════════════════════════════

section("Compression Ablation: SVD/FIFO/STOP  (Exps 421, 422, 423)")
accs421,_ = load_exp("421.csv", methods_filter=["ifopng"])
accs422,_ = load_exp("422.csv", methods_filter=["ifopng"])
accs423,_ = load_exp("423.csv", methods_filter=["ifopng"])

check("SVD  (thesis ~58.7%)", mean_of(accs421,"ifopng"), 0.587, computed_n=n_of(accs421,"ifopng"))
check("FIFO (thesis ~58.8%)", mean_of(accs422,"ifopng"), 0.588, computed_n=n_of(accs422,"ifopng"))
check("STOP (thesis ~58.9%)", mean_of(accs423,"ifopng"), 0.589, computed_n=n_of(accs423,"ifopng"))

svd_v  = list(accs421.get("ifopng",{}).values())
fifo_v = list(accs422.get("ifopng",{}).values())
stop_v = list(accs423.get("ifopng",{}).values())
if svd_v and fifo_v and stop_v:
    f,fp = stats.f_oneway(svd_v, fifo_v, stop_v)
    sig  = "non-significant ✓" if fp > 0.05 else "SIGNIFICANT — thesis claims indistinguishable"
    print(f"  INFO  One-way ANOVA: F={f:.2f} p={fp:.3f} → {sig}")

# ═══════════════════════════════════════════════════════════════════════════════
# E17: Permuted-MNIST task count in Sub-RQ3 design section
# ═══════════════════════════════════════════════════════════════════════════════

section("E17: Permuted-MNIST task count (design says 20T, data is 5T)")
df401 = pd.read_csv(DATA_DIR / "401.csv")
tc_vals = {safe_parse(r["config"]).get("num_tasks","?")
           for _, r in df401.iterrows()}
print(f"  INFO  Exp401 num_tasks: {tc_vals}  "
      f"(experiment_design.tex L176/180 claims 20 — WRONG)")

found = any(
    safe_parse(r["config"]).get("methods",["?"])[0] if isinstance(
        safe_parse(r["config"]).get("methods"),list) else
    safe_parse(r["config"]).get("methods","?") == "fopng"
    and safe_parse(r["config"]).get("task") == "permuted_mnist"
    and safe_parse(r["config"]).get("num_tasks") == 20
    for fname in sorted(DATA_DIR.glob("*.csv"))
    for _, r in pd.read_csv(fname).iterrows()
)
print(f"  {'PASS ' if not found else 'INFO '} "
      f"20-task FOPNG permuted run: {'NOT FOUND — E17 confirmed' if not found else 'found'}")

# ═══════════════════════════════════════════════════════════════════════════════
# E19: EWC λ in hyperparameter table
# ═══════════════════════════════════════════════════════════════════════════════

section("E19: EWC λ values  (hyper_parameters.tex Table tab:ewc_lambda)")
EWC_CHECK = {
    "401.csv": ("Permuted-MNIST",  10.0,   "10?"),
    "402.csv": ("Split-MNIST MH", 400.0,  "0.0005?"),
    "403.csv": ("Split-MNIST SH", 400.0,  "0.0005?"),
    "404.csv": ("Split-CIFAR10",   50.0,  "10or50?"),
}
for fname, (bench, expected_lam, table_claim) in EWC_CHECK.items():
    df = pd.read_csv(DATA_DIR / fname)
    ewc_lams = set()
    for _, row in df.iterrows():
        c = safe_parse(row["config"])
        m = c.get("methods",["?"])
        if isinstance(m,list): m = m[0]
        if m == "ewc":
            ewc_lams.add(c.get("lam","?"))
    correct = expected_lam in ewc_lams
    tag  = "PASS " if correct and table_claim == str(int(expected_lam)) else \
           "FAIL " if not correct else "WARN "
    print(f"  {'PASS ' if correct else 'FAIL '} "
          f"{bench}: actual EWC λ={ewc_lams}  "
          f"table says '{table_claim}'  "
          f"{'✓ correct' if correct else '← WRONG — table must be updated'}")

# ═══════════════════════════════════════════════════════════════════════════════
# λ DISCREPANCY: Sub-RQ4 vs Compression ablation
# ═══════════════════════════════════════════════════════════════════════════════

section("λ discrepancy: Sub-RQ4 (10⁻²) vs Compression (10⁻³)")
for exp_id, fname, method in [(414,"414.csv","ifopng_ema"),(415,"415.csv","ifopng"),
                               (421,"421.csv","ifopng"),(422,"422.csv","ifopng"),
                               (423,"423.csv","ifopng")]:
    df = pd.read_csv(DATA_DIR / fname)
    lams = set()
    for _, row in df.iterrows():
        c = safe_parse(row["config"])
        m = c.get("methods",["?"])
        if isinstance(m,list): m = m[0]
        if m == method: lams.add(c.get("lam","?"))
    print(f"  INFO  Exp{exp_id} ({method}): λ={lams}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
total = PASS_COUNT + WARN_COUNT + FAIL_COUNT
print(f"""
  PASS : {PASS_COUNT:3d}  ({PASS_COUNT/max(total,1)*100:.0f}%)  — within 0.15pp
  WARN : {WARN_COUNT:3d}  ({WARN_COUNT/max(total,1)*100:.0f}%)  — within 0.5pp (seed-subset or rounding)
  FAIL : {FAIL_COUNT:3d}  ({FAIL_COUNT/max(total,1)*100:.0f}%)  — material discrepancy
  Total: {total:3d} numeric checks
""")
if FAIL_COUNT == 0:
    print("  ✓ All numbers validated or within rounding tolerance.")
else:
    print("  ✗ Review FAIL lines above before submission.")