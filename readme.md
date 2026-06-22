<div align="center">

<br/>

```
██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ███████╗██╗███████╗██╗  ██╗███████╗██████╗
██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝██║██╔════╝██║  ██║██╔════╝██╔══██╗
███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝█████╗  ██║███████╗███████║█████╗  ██████╔╝
██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██╔══╝  ██║╚════██║██╔══██║██╔══╝  ██╔══██╗
██║  ██║   ██║   ██║     ███████╗██║  ██║██║     ██║███████║██║  ██║███████╗██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
```

### *Gradient Projection Methods for Continual Learning in Hypernetworks*

<br/>

**Jakub Michałowski** · Bachelor's Thesis · Tilburg University
Department of Cognitive Science and Artificial Intelligence
Supervised by Dr. Giacomo Spigler

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_&_Biases-tracked-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

<br/>

</div>

---

## ◈ What this is

A continual-learning system needs to learn a sequence of tasks without forgetting
earlier ones. Two families of methods attack this problem from very different angles:

- **Gradient projection** methods (OGD, ONG, FNG, FOPNG) constrain *how the weights move*,
  projecting each update away from directions that would disturb earlier tasks.
- **Hypernetworks** attack it from the *architecture* side: a small network generates the
  weights of a target network from a per-task embedding, so the storage cost of "remembering"
  is decoupled from the number of parameters in the target.

This repository is the implementation behind my thesis, which asks a question that — to my
knowledge — had not been studied directly:

> **Can Fisher-based gradient projection methods be constructively integrated into a
> chunked hypernetwork, or do the two mechanisms interfere?**

The honest short answer the thesis arrives at is **mostly "they interfere"** — and the
interesting part is *why*. The chunked weight-generation process distorts the geometry that
projection methods rely on, and fixing that distortion is one of the two technical
contributions here. The other is a new projection variant, **iFOPNG**, that builds
parameter *inertia* directly into the metric rather than into a penalty term.

> ⚠️ **A note on framing.** An earlier version of this README reported a single
> preliminary run in which FOPNG reached ~97% on Split-MNIST. That number was an early,
> cherry-picked artefact and **does not survive a proper multi-seed, multiple-comparison-corrected
> evaluation.** The results section below reflects the finished thesis, not that prototype.

<br/>

---

## ◈ Research questions

**Main RQ.** Can Fisher-based gradient projection methods be integrated into chunked
hypernetwork architectures for continual learning, and how do they compare to standard
baselines in that setting?

Decomposed into four sub-questions, each mapped to a set of experiments:

| Sub-RQ | Question |
|--------|----------|
| **RQ1** | Do projection methods and hypernetwork architectures **synergise or conflict** — standalone target network vs. chunked hypernetwork? |
| **RQ2** | Does the proposed **projection-scoped normalisation** resolve the chunk-induced conditioning pathology in the Fisher/projection machinery? |
| **RQ3** | Does **iFOPNG** (combined-Fisher inertia metric) improve over **FOPNG** (single-task Fisher metric)? |
| **RQ4** | Does **MAX** or **EMA** Fisher accumulation across tasks retain old tasks better? |

<br/>

---

## ◈ What I actually found

Stated plainly, with the caveats that matter:

- **In the hypernetwork setting, the simple baselines win.** Adam and EWC form a
  statistically indistinguishable top tier and **significantly outperform every projection
  method** (e.g. Adam vs. iFOPNG, Holm-corrected *p* ≈ 0.005). Projection methods do not
  earn their keep once a hypernetwork is in the loop.
- **iFOPNG is the best of the projection family, but the win is fragile.** iFOPNG improves
  on FOPNG, but **iFOPNG vs. SGD does not survive Holm–Bonferroni correction** — so this is
  reported as a trend, not a confirmed effect.
- **The chunk-conditioning pathology is real and quantifiable.** With *C* chunks, the
  backward pass scales gradients by ~*C* and the diagonal Fisher by ~*C²*, so the projection
  matrix *A* inflates by roughly *C⁴*. This pushes *A* out of the usable floating-point range.
  Note that κ(*A*) is **scale-invariant** — so the proposed normalisation is a
  *float-precision* fix that restores a workable regime, **not** a reduction of the condition
  number per se. This is the core RQ2 finding.
- **MAX beats EMA** for cross-task Fisher accumulation by a significant margin (RQ4).

These are the claims the thesis defends; exact per-seed numbers, significance tables, and
trajectory plots live in W&B and in the thesis figures (see *Experiment tracking* below).

<br/>

---

## ◈ The two contributions

**① iFOPNG — "inertia" Fisher-Orthogonal Projected Natural Gradient.**
FOPNG preconditions the projected update with the *new* task's Fisher, $\hat F_\text{new}$.
iFOPNG instead uses a **combined metric**

$$F_c = \hat F_\text{new} + \hat F_\text{old}$$

so that parameters that were important to previous tasks acquire elevated metric mass and
*geometrically resist displacement* — no explicit EWC-style penalty term required. The
contrast with EWC is the point: EWC enforces inertia through a loss penalty; iFOPNG bakes it
into the ambient metric. Gradient storage (the `PreFisher` path) is identical to FOPNG; the
**only** difference is the inverse metric used at update time ($F_c^{-1}$ instead of
$\hat F_\text{new}^{-1}$).

> 🏷️ **Naming.** The method is called **iFOPNG** in the thesis (the *i* is for *inertia*).
> For backwards compatibility the CLI flag and the class are still named `efopng` / `eFOPNG`
> — see the flag-mapping table below. They are the same method.

**② Projection-scoped normalisation.**
To counter the *C⁴* inflation above, the gradient basis is **QR-orthonormalised at insertion
time** and the Fisher is rescaled by its **max entry** before the projection matrix is
formed. This is applied locally, scoped to the projection step (hence "projection-scoped"),
and restores numerically stable matrix inversion in the chunked setting. Enabled with
`--normalize`.

<br/>

---

## ◈ Methods implemented

**Baselines**

| Flag | Method | Idea |
|------|--------|------|
| `sgd` | SGD | Naive sequential SGD — lower bound |
| `adam` | Adam | Naive sequential Adam — strong, forgetting-prone baseline |
| `ewc` | EWC | Diagonal-Fisher penalty on important weights (Kirkpatrick et al., 2017) |

**Projection family** (all share the `OP` base class in `optimizers/projections.py`)

| Flag | Method | Metric / projection |
|------|--------|---------------------|
| `ogd` | OGD | Euclidean orthogonal projection, no Fisher (Farajtabar et al., 2020) |
| `ong` | ONG | Orthogonal natural gradient |
| `fng` | FNG | Natural gradient under $\hat F_\text{new}$, no projection |
| `fopng` | FOPNG | Fisher-orthogonal projected natural gradient (Garg et al., 2026) |
| `efopng` | **iFOPNG** *(ours)* | Same as FOPNG but combined-Fisher inertia metric $F_c$ |

**Variants** (storage / accumulation ablations)

| Flag | Variant |
|------|---------|
| `fopng_prefisher` | FOPNG with Fisher pre-multiplied into stored gradients |
| `efopng_prefisher` | iFOPNG with pre-Fisher gradient storage |
| `efopng_ema` | iFOPNG with EMA (rather than MAX) Fisher accumulation — RQ4 ablation |

<br/>

---

## ◈ Benchmarks & settings

Every benchmark is run in **both** settings, which is what RQ1 turns on:

- **Standalone** — projection method applied directly to a target network (`--model TargetNetwork`)
- **Hypernetwork** — projection method applied to the shared hypernetwork parameters that
  generate the target weights (`--model HyperNetwork`)

| Benchmark | Flag | Tasks | Heads |
|-----------|------|-------|-------|
| Permuted-MNIST | `permuted_mnist` | configurable | single |
| Split-MNIST (single-head) | `split_mnist_sh` | 5 binary | single |
| Split-MNIST (multi-head) | `split_mnist_mh` | 5 binary | multi |
| Split-CIFAR-10 | `split_cifar10` | 5 | multi |
| Split-CIFAR-100 | `split_cifar100` | 20 | multi |

<br/>

---

## ◈ Repository structure

```
HyperFisher/
│
├── main.py                  ← Entry point; full experiment CLI
│
├── fisher.py                ← DiagonalFisherEstimator (MAX/EMA accumulation, clipping, normalisation)
├── gradient.py              ← Gradient memory + collectors (raw / pre-Fisher); QR at insertion
├── utils.py                 ← Shared utilities (grad vectors, BWT, evaluation, plotting)
│
├── models/
│   ├── hyper_network.py     ← Chunked HyperNetwork + HyperRegulizer
│   ├── mlp.py               ← MLP target network
│   └── cnn.py               ← CNN target network (CIFAR)
│
├── optimizers/
│   ├── projections.py       ← OP base class + OGD / ONG / FNG / FOPNG / iFOPNG (+ variants)
│   ├── ewc.py               ← EWC baseline
│   └── vanilla.py           ← SGD / Adam baselines
│
├── tasks/
│   ├── permuted_mnist.py
│   ├── split_mnist_sh.py
│   ├── split_mnist_mh.py
│   ├── split_cifar10.py
│   └── split_cifar100.py
│
├── config_*.sh              ← SLURM launch scripts, one per thesis configuration (1–23)
│
└── plotting/                ← Figure generators (rq1_*, rq3, rq5, trajectories, …)
```

> Each task module exposes a `TaskGenerator` with `.config`, `.target_network`,
> `.generate(...)` and `.solo_target(...)`, so `main.py` can load any benchmark by name
> via `importlib`.

<br/>

---

## ◈ Installation

```bash
git clone --recurse-submodules <repo-url>
cd HyperFisher
conda create -n venv python=3.10 && conda activate venv
pip install torch torchvision wandb matplotlib numpy tqdm
```

A CUDA GPU is recommended; the code falls back to CPU automatically. Datasets download on
first run.

<br/>

---

## ◈ Usage

The entry point is `main.py`. Pick a benchmark, a model setting, and one or more methods to
run sequentially in the same process.

```bash
# Split-MNIST (single-head), hypernetwork, baselines vs. projection family
python main.py \
  --task split_mnist_sh \
  --model HyperNetwork \
  --methods adam ewc fopng efopng \
  --epochs 10 --lr 1e-3 \
  --normalize

# Standalone target network — the RQ1 "no hypernetwork" comparison point
python main.py \
  --task split_cifar10 \
  --model TargetNetwork \
  --methods sgd efopng \
  --no-regulizer

# RQ4 ablation: MAX vs. EMA Fisher accumulation
python main.py \
  --task permuted_mnist \
  --model HyperNetwork \
  --methods efopng efopng_ema
```

On the cluster, the per-configuration SLURM scripts wrap these calls:

```bash
sbatch config_8.sh     # Split-CIFAR10, standard HN (RQ1b / RQ2)
sbatch config_4.sh     # Split-CIFAR10, standalone (RQ1 / RQ3 / RQ4)
```

<br/>

**Key CLI arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | *required* | `permuted_mnist` · `split_mnist_sh` · `split_mnist_mh` · `split_cifar10` · `split_cifar100` |
| `--model` | `HyperNetwork` | `HyperNetwork` or `TargetNetwork` |
| `--methods` | `fopng adam` | Any of: `sgd adam ewc ogd ong fng fopng efopng` (+ `*_prefisher`, `efopng_ema`) |
| `--seed` | `1000` | Random seed (canonical set: `42 111 811 1234 2137`) |
| `--epochs` | `10` | Epochs per task |
| `--lr` | `1e-3` | Learning rate |
| `--first_task_lr` / `--first_task_opt` | `1e-3` / — | First-task LR and optimiser (see reproducibility note) |
| `--lam` | `1e-3` | Damping (λI) on the projection / Fisher inverse |
| `--alpha` | `0.3` | EMA coefficient (for `efopng_ema`) |
| `--normalize` | off | **Projection-scoped normalisation** (QR + max-entry Fisher scaling) |
| `--fisher_samples` | `1024` | Samples for diagonal Fisher estimation |
| `--grads_per_task` / `--max_directions` | `40` / `80` | Gradient-memory size per task / hard cap |
| `--hyper_hidden_dim` | `16` | Hypernetwork bottleneck width |
| `--task_embedding_dim` / `--chunk_embedding_dim` | `4` / `10` | Task / chunk embedding sizes |
| `--chunk_size` | `1000` | Target-weight chunk size |
| `--regulizer` / `--no-regulizer` | on | von Oswald output regulariser (HN only) |
| `--beta` | `0.1` | Regulariser strength |

<br/>

---

## ◈ Reproducibility notes

A few honest caveats that the thesis discusses and that anyone re-running this should know:

- **First-task learning rate.** Garg et al.'s stated SGD setting (1e-5 on Split-MNIST) yields
  only ~21% first-task accuracy in our reproduction. For internal comparability, **all**
  methods initialise the first task with Adam at 1e-3; the projection machinery only engages
  from task 2 onward.
- **Dropout is disabled in the hypernetwork setting.** `functional_call` propagates training
  mode into the generated target network, so dropout masks contaminate the gradient signal
  across chunks. It is removed in all HN configurations.
- **Seed exclusion is disclosed, not hidden.** Most experiments use
  `{42, 111, 811, 1234, 2137}`. Split-MNIST HN uses `{42, 111, 314, 811, 1234}`: seed 2137
  is excluded because projection-method initialisation fails on the `d_h=8` bottleneck — and
  it is documented rather than silently swapped, since the same seed behaves normally for
  every other method.

<br/>

---

## ◈ Experiment tracking

All runs log to **Weights & Biases** under `michalowski-jb-tilburg-university/HyperFisher`,
grouped by task and by model setting. Each thesis experiment has a numeric ID; the exported
CSVs (`401.csv`–`423.csv`, plus `701`–`703`) are the frozen snapshots used to generate the
figures. The `plotting/` scripts (`rq1_*`, `rq3.py`, `rq5.py`, `trajectories.py`,
`generate_1_2_5_6_12_13.py`, …) read those CSVs / the W&B API and emit both `.pdf` and `.png`.

W&B summary layout used by the plotters:

```python
summary["best/results"]["acc"]["1".."5"]   # final per-task accuracy
summary["best/results"]["bwt"]             # backward transfer
summary["best/average_accuracy"]           # mean final accuracy
```

<br/>

---

## ◈ References

```
Garg, I., Kolhe, N., Peng, A., & Gopalam, R. (2026).
  Fisher-orthogonal projected natural gradient descent for continual learning.

von Oswald, J., Henning, C., Grewe, B. F., & Sacramento, J. (2020).
  Continual learning with hypernetworks. ICLR.

Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020).
  Orthogonal gradient descent for continual learning. AISTATS.

Kirkpatrick, J., et al. (2017).
  Overcoming catastrophic forgetting in neural networks. PNAS, 114(13), 3521–3526.

Chang, O., Flokas, L., & Lipson, H. (2020).
  Principled weight initialisation for hypernetworks. ICLR.

Ha, D., Dai, A., & Le, Q. V. (2017).
  HyperNetworks. ICLR.
```

> `FOPNG/`, `fop/`, and `hypercl/` (if present as submodules) are external reference
> codebases included for replication — not part of this thesis's contribution.

<br/>

---

<div align="center">

*Tilburg University · 2025–2026*

</div>