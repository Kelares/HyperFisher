#!/bin/bash
# ==============================================================================
# Split-CIFAR100 — Replication of Garg et al. (2026) Table 1 + eFOPNG
# ==============================================================================
# Architecture : MultiHeadCNN100 (same backbone as CIFAR10, 10-class heads)
#                Multi-head, 10 outputs per task, labels remapped to 0…9
#                No data augmentation — ToTensor + Normalize only
# Hyperparams  : Matched exactly to Table 1 (Split-CIFAR100 column)
# eFOPNG       : Same hyperparameters as FOPNG (novel contribution)
# ONG          : Not in paper; uses OGD settings as closest reference
# Seeds        : 3  (paper uses 5; reduce if time is short)
#
# Table 1 reference (Split-CIFAR100):
#   Method  lr      lam    grads  max_dirs  fisher  epochs
#   Adam    1e-4    —      —      —         —       10
#   SGD     5e-3    —      —      —         —       10
#   EWC     1e-2    50     —      —         1024    10
#   FNG     5e-3    1e-3   80     —         1024    10
#   OGD     1e-2    —      80     800       —       10
#   FOPNG   5e-3    1e-3   80     800       1024    10
#   eFOPNG  5e-3    1e-3   80     800       1024    10  ← same as FOPNG
# ==============================================================================

#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm/%j.out # STDOUT
#SBATCH -e slurm/%j.%N.err # STDERR
#SBATCH --gres=gpu:1
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi
cd ~/SSM_benchmark/
conda activate venv


TASK="split_cifar100"
DEVICE="gpu"
MODEL="TargetNetwork"
SEEDS=(42 1234 811)
EPOCHS=10
BATCH=10
GRADS=80
MAX_DIRS=800
FISHER=1024

# ── Per-method learning rates (Table 1, Split-CIFAR100) ──────────────────────
declare -A LR
LR["adam"]="1e-4"
LR["sgd"]="5e-3"
LR["ewc"]="1e-2"
LR["fng"]="5e-3"
LR["ogd"]="1e-2"
LR["ong"]="1e-2"
LR["fopng"]="5e-3"
LR["efopng"]="5e-3"

# ── Per-method lambda (Table 1; 0 = flag omitted) ────────────────────────────
declare -A LAM
LAM["adam"]="0"
LAM["sgd"]="0"
LAM["ewc"]="50"
LAM["fng"]="1e-3"
LAM["ogd"]="0"
LAM["ong"]="0"
LAM["fopng"]="1e-3"
LAM["efopng"]="1e-3"

ALL_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")

echo "======================================================================"
echo " Split-CIFAR100 — FOPNG Table 1 replication + eFOPNG"
echo " Batch=$BATCH  Epochs=$EPOCHS  Fisher=$FISHER  Seeds=${SEEDS[*]}"
echo "======================================================================"

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "  method=$METHOD  lr=${LR[$METHOD]}  lam=${LAM[$METHOD]}  seed=$SEED"

        ARGS=(
            --task=$TASK
            --model=$MODEL
            --methods=$METHOD
            --no-regulizer
            --grads_per_task=$GRADS
            --max_directions=$MAX_DIRS
            --fisher_samples=$FISHER
            --device_mode=$DEVICE
            --lr=${LR[$METHOD]}
            --max_epochs=$EPOCHS
            --batch_size=$BATCH
            --seed=$SEED
            --lam=${LAM[$METHOD]}
        )
        python main.py "${ARGS[@]}"

        echo "  Done: $METHOD seed=$SEED"
    done
done

echo ""
echo "======================================================================"
echo " ALL RUNS COMPLETE"
echo "======================================================================"