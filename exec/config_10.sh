#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.%N.err
#SBATCH --gres=gpu:1

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

cd ~/HyperFisher/
conda activate venv

# ==============================================================================
# Full Thesis Experiment Suite — All 13 Configurations
# ==============================================================================
#
# CONFIG  BENCHMARK                  SETTING          SUB-RQ   STATUS
# ──────  ─────────────────────────  ───────────────  ───────  ──────────
# 1       Permuted-MNIST             Standalone       4        TODO
# 2       Split-MNIST MH             Standalone       4, 1a    TODO
# 3       Split-MNIST SH             Standalone       4        TODO
# 4       Split-CIFAR10 MH           Standalone AdamW 4,1a,3B  TODO
# 5       Split-CIFAR10 MH           Standalone Adam  3A       TODO
# 6       Split-CIFAR100 MH          Standalone       4, 1a    TODO
# 7       Split-MNIST SH             Suffocated HN    1b       DONE (5 seeds)
# 8       Split-CIFAR10              Standard HN      1b, 2C3  DONE (3 seeds)
# 9       Split-CIFAR10              HN no-norm       2C1      TODO
# 10      Split-CIFAR10              HN grad-only     2C2      TODO
# 11      Split-CIFAR100             Standard HN      1b       TODO
# 12      Split-MNIST SH (d_h=4)     Prelim sweep     App.     TODO
# 13      Split-MNIST SH (d_h=16)    Prelim sweep     App.     TODO
#
# Hyperparameters for configs 1-6: Garg et al. (2026) Table 1 exactly.
# Hyperparameters for configs 7-13: custom (justified in Methods section).
# ==============================================================================
# ("adam" "ifopng" "fopng" "ogd" "ong" "fng" "ewc" "sgd")
#           GRADIENT ONLY NORMALIZATION. DEMANDS A TWEAK      ############

DEVICE="gpu"
SEEDS_3=(42 1234 811)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 10 — Split-CIFAR10 Standard HN, GRADIENT normalization only
# (Sub-RQ2 Condition 2)
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 10: Split-CIFAR10 HN — Gradient-only normalization (Sub-RQ2 Cond 2) ==="

for SEED in "${SEEDS_3[@]}"; do
    echo "--> C10 ifopng seed=$SEED"
    python main.py \
        --task=split_cifar10 \
        --methods=ifopng \
        --regulizer \
        --hyper_hidden_dim=64 \
        --task_embedding_dim=32 \
        --chunk_embedding_dim=32 \
        --chunk_size=256 \
        --grads_per_task=80 --max_directions=400 \
        --fisher_samples=1024 \
        --lr=1e-3 --max_epochs=50 --batch_size=64 \
        --lam=1e-3 \
        --first_task_opt=adamw --first_task_lr=1e-3 \
        --normalize_gradients_only \
        --device_mode=$DEVICE --seed=$SEED --experiment_id=410
done
