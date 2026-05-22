#!/bin/bash

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
cd ~/HyperFisher/
conda activate venv

# ==============================================================================
# Permuted-MNIST — Replication of Garg et al. (2026) Table 1 + eFOPNG
# ==============================================================================
# Architecture : MLP 784→100→100→10, single-head, no normalisation
# Hyperparams  : Matched exactly to Table 1 (Permuted-MNIST column)
# eFOPNG       : Same hyperparameters as FOPNG (novel contribution)
# ONG          : Not in paper; uses OGD settings as closest reference
# Seeds        : 3  (paper uses 5; reduce if time is short)
#
# Table 1 reference (Permuted-MNIST):
#   Method  lr      lam    grads  fisher
#   Adam    1e-4    —      —      —
#   SGD     5e-3    —      —      —
#   EWC     1e-2    10     —      full
#   FNG     1e-3    1e-3   80     full
#   OGD     5e-3    —      80     —
#   FOPNG   1e-4    1e-2   80     full
#   eFOPNG  1e-4    1e-2   80     full   ← same as FOPNG
# ==============================================================================

TASK="permuted_mnist"
DEVICE="gpu"
MODEL="TargetNetwork"
SEEDS=(42 1234 811)
EPOCHS=5
BATCH=10
GRADS=80
MAX_DIRS=400
FISHER=60000      # "full" = entire MNIST training set (60K samples)

# ── Per-method learning rates (Table 1, Permuted-MNIST) ──────────────────────
declare -A LR
LR["adam"]="1e-4"
LR["sgd"]="5e-3"
LR["ewc"]="1e-2"
LR["fng"]="1e-3"
LR["ogd"]="5e-3"
LR["ong"]="5e-3"
LR["fopng"]="1e-4"
LR["efopng"]="1e-4"

# ── Per-method lambda (Table 1; 0 = flag omitted) ────────────────────────────
declare -A LAM
LAM["adam"]="0"
LAM["sgd"]="0"
LAM["ewc"]="10"
LAM["fng"]="1e-3"
LAM["ogd"]="0"
LAM["ong"]="0"
LAM["fopng"]="1e-2"
LAM["efopng"]="1e-2"

ALL_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")

echo "======================================================================"
echo " Permuted-MNIST — FOPNG Table 1 replication + eFOPNG"
echo " Batch=$BATCH  Epochs=$EPOCHS  Fisher=full($FISHER)  Seeds=${SEEDS[*]}"
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