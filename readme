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

### *Fisher-Orthogonal Parameter Manifolds*
### *Enforcing Hard Constraints in Hypernetwork-based Continual Learning*

<br/>

**Jakub Michałowski** · Thesis Repository  
Department of Cognitive Science and Artificial Intelligence · Tilburg University

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_&_Biases-tracked-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

<br/>

</div>

---

## ◈ What is this?

Can a neural network learn a new task **without forgetting the old ones**?

This repository contains my thesis implementation of **FOPNG** (*Fisher-Orthogonal Projected Natural Gradient Descent*) applied to **hypernetworks** for continual learning. The core idea: instead of operating in flat Euclidean space like most methods, FOPNG respects the true Riemannian geometry of the parameter space — using the Fisher information matrix to define what "changing old task outputs" actually means, then enforcing a hard constraint that new updates cannot do it.

The hypernetwork wrapper keeps the Fisher matrix and gradient memory **compact and task-count-independent**, making the approach tractable even for larger target networks.

<br/>

---

## ◈ Repository Map

```
./
│
├── 🧠  HyperFisher/          ← Main project (my code)
│
├── 🏋️  gym/                  ← Side project: SSMs vs Transformers on
│                               memory-intensive RL benchmarks
│
├── 📄  misc/                 ← Thesis writing · poster · proposal · residuals
│
├── 📦  FOPNG/                ← Reference: Garg et al. (2026) original implementation
├── 📦  fop/                  ← Reference: related paper codebase
├── 📦  hypercl/              ← Reference: von Oswald et al. (2020) hypernetwork CL
│
└── 🧪  toy_playground/       ← Scratch experiments and quick prototypes
```

> **Note:** `FOPNG/`, `fop/`, and `hypercl/` are external codebases included for replication and reference — not my own work.

<br/>

---

## ◈ HyperFisher — Main Project

```
HyperFisher/
│
├── main.py               ← Entry point — full CLI for all experiments
├── main.sh               ← Example run scripts
├── hyper_network.py      ← HyperNetwork: generates target weights from task embeddings
├── mlp_base.py           ← Plain MLP target network
├── utils.py              ← Shared utilities
│
├── optimizers/
│   ├── fopng.py          ← FOPNG — Fisher-Orthogonal Projected Natural Gradient
│   ├── ewc.py            ← EWC — Elastic Weight Consolidation baseline
│   └── adam.py           ← Adam baseline (naive, no forgetting protection)
│
├── tasks/
│   ├── split_mnist.py    ← Split-MNIST: 5 binary tasks (0v1, 2v3, 4v5, 6v7, 8v9)
│   ├── permuted_mnist.py ← Permuted-MNIST: 10+ random permutation tasks
│   └── split_cifar10.py  ← Split-CIFAR-10: 5 tasks on natural images
│
├── visualizations/       ← Trajectory plots and per-task accuracy graphs
├── sweep.yaml            ← W&B hyperparameter sweep config
│
├── data/                 ← Auto-populated on first run
└── wandb/                ← W&B run logs (auto-populated)
```

<br/>

---

## ◈ The Method

**FOPNG** combines two ideas applied to the compact hypernetwork parameter set φ:

<br/>

**① Natural Gradient** — Standard gradient descent treats all parameter directions equally. The Fisher information matrix $\mathcal{F}_\theta$ captures how sensitively the model's output *distribution* responds to each parameter. Natural gradient descent preconditions updates with $\mathcal{F}_\theta^{-1}$, making steps equal-sized in *distribution space* rather than Euclidean space:

$$\Delta\theta_\text{nat} = -\eta \, \mathcal{F}_\theta^{-1} \nabla_\theta \mathcal{L}$$

**② Orthogonal Projection** — After each task $k$, its gradient $g_k$ is stored in a memory matrix $G = [g_1 \mid \cdots \mid g_{t-1}]$. New updates are projected onto the **Fisher-orthogonal complement** of $G$ — directions guaranteed not to alter prior task outputs:

$$\Delta\theta = -\eta \left[ I - G(G^\top \mathcal{F}_\theta G)^{-1} G^\top \mathcal{F}_\theta \right] \mathcal{F}_\theta^{-1} \nabla_\theta \mathcal{L}_t$$

Applied to the **hypernetwork** φ (rather than the full target network), both the Fisher matrix and gradient memory stay compact and task-count-independent.

<br/>

---

## ◈ Baselines

| Method | Type | Key idea |
|--------|------|----------|
| **Adam** | Naive | No forgetting protection — establishes lower bound |
| **EWC** | Regularization | Diagonal Fisher penalty on important weights, Euclidean space |
| **OGD** | Projection | Euclidean orthogonal gradient projection — closest analogue to FOPNG |
| **FNG** | Natural gradient | Natural gradient without orthogonal projection |

<br/>

---

## ◈ Benchmarks

| Benchmark | Tasks | Input | Notes |
|-----------|-------|-------|-------|
| **Split-MNIST** | 5 binary | 784-dim | Entry-level sequential learning |
| **Permuted-MNIST** | 10+ | 784-dim | Long-horizon retention test |
| **Split-CIFAR-10** | 5 | 3072-dim | Natural images, higher complexity |
| **Split-CIFAR-100** | 20 | 3072-dim | 100 classes, hardest benchmark |

<br/>

---

## ◈ Usage

```bash
cd HyperFisher

# Split-MNIST with all three methods
python main.py \
  --task split_mnist \
  --methods fopng ewc adam \
  --model HyperNetwork \
  --epochs 5 \
  --lr 1e-3 \
  --embedding_dim 4

# Split-CIFAR-10, FOPNG only, more gradient memory
python main.py \
  --task split_cifar10 \
  --methods fopng \
  --model HyperNetwork \
  --epochs 10 \
  --grads_per_task 40 \
  --max_directions 200

# Run a W&B hyperparameter sweep
bash sweep.sh
```

<br/>

**Key CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | *required* | `split_mnist` · `permuted_mnist` · `split_cifar10` |
| `--methods` | `fopng adam` | Any of: `fopng` `ewc` `adam` `ogd` `fng` |
| `--model` | `HyperNetwork` | `HyperNetwork` or `MLP` |
| `--epochs` | `10` | Epochs per task |
| `--lr` | `1e-3` | Learning rate |
| `--lam` | `1e-3` | EWC regularization strength |
| `--grads_per_task` | `40` | Gradient directions stored per task |
| `--max_directions` | `80` | Hard cap on memory matrix columns |
| `--embedding_dim` | `4` | Task embedding dimension |
| `--hyper_hidden_dim` | `16` | Hypernetwork bottleneck width |

<br/>

---

## ◈ Preliminary Results

> Split-MNIST · Full Hypernetwork · 5 sequential binary tasks

After training on all 5 tasks, final per-task accuracy:

| Method | T1 | T2 | T3 | T4 | T5 | **Avg** |
|--------|----|----|----|----|----|----|
| Adam (baseline) | 99% | 50% | 97% | 47% | 98% | 78% |
| EWC | 52% | 56% | 48% | 98% | 97% | 70% |
| **FOPNG (ours)** | **100%** | **97%** | **93%** | **99%** | **97%** | **97%** |

FOPNG successfully maintains near-perfect accuracy across all tasks with minimal forgetting. Adam collapses on previously learned tasks within a few update steps. EWC partially mitigates forgetting but degrades significantly under longer task sequences.

<br/>

---

## ◈ Installation

```bash
git clone --recurse-submodules <repo-url>
cd HyperFisher
pip install torch torchvision wandb
```

CUDA is recommended. The training loop detects GPU automatically and falls back to CPU.

Datasets are downloaded automatically on first run into `HyperFisher/data/`.

<br/>

---

## ◈ References

```
Garg, I., Kolhe, N., Peng, A., & Gopalam, R. (2026).
  Fisher-orthogonal projected natural gradient descent for continual learning.

von Oswald, J., Henning, C., Grewe, B. F., & Sacramento, J. (2020).
  Continual learning with hypernetworks. ICLR.

Kirkpatrick, J., et al. (2017).
  Overcoming catastrophic forgetting in neural networks. PNAS, 114(13), 3521–3526.

Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020).
  Orthogonal gradient descent for continual learning. AISTATS.

Ha, D., Dai, A., & Le, Q. V. (2017).
  HyperNetworks. ICLR.
```

<br/>

---

<div align="center">

*Tilburg University · 2025–2026*

</div>