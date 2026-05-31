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
# Split-MNIST — Replication of Garg et al. (2026) Table 1 + iFOPNG
# ==============================================================================
# Architecture : MLP 784→100→100→10, single-head, labels remapped to {0,1}
# Hyperparams  : Matched exactly to Table 1 (Split-MNIST column)
# iFOPNG       : Same hyperparameters as FOPNG (novel contribution)
# ONG          : Not in paper; uses OGD settings as closest reference
# Seeds        : 3  (paper uses 5; reduce if time is short)
#
# Table 1 reference (Split-MNIST):
#   Method  lr      lam    grads  fisher
#   Adam    1e-5    —      —      —
#   SGD     5e-4    —      —      —
#   EWC     5e-4    400    —      full
#   FNG     1e-3    1e-3   80     full
#   OGD     5e-4    —      80     —
#   FOPNG   1e-5    5e-4   80     full
#   iFOPNG  1e-5    5e-4   80     full   ← same as FOPNG
# ==============================================================================

TASK="split_mnist_mh"
DEVICE="gpu"
MODEL="TargetNetwork"
SEEDS=(42) #1234 811)
EPOCHS=5
BATCH=10
GRADS=80
MAX_DIRS=400
FISHER=1200      # "full" in paper ≈ 12K samples per 2-class MNIST task

# ── Per-method learning rates (Table 1) ──────────────────────────────────────
declare -A LR
LR["adam"]="1e-5"
LR["sgd"]="5e-4"
LR["ewc"]="5e-4"
LR["fng"]="1e-3"
LR["ogd"]="5e-4"
LR["ong"]="5e-4"
LR["fopng"]="1e-5"
LR["ifopng"]="1e-5"
LR["ifopng_prefisher"]="1e-5"

# ── Per-method lambda (Table 1; 0 = flag omitted) ────────────────────────────
declare -A LAM
LAM["adam"]="0"
LAM["sgd"]="0"
LAM["ewc"]="400"
LAM["fng"]="1e-3"
LAM["ogd"]="0"
LAM["ong"]="0"
LAM["fopng"]="5e-4"
LAM["ifopng"]="5e-4"
LAM["ifopng_prefisher"]="5e-4"


ALL_METHODS=("ifopng_prefisher" "ifopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")

echo "======================================================================"
echo " Split-MNIST — FOPNG Table 1 replication + iFOPNG"
echo " Batch=$BATCH  Epochs=$EPOCHS  Fisher≈full($FISHER)  Seeds=${SEEDS[*]}"
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