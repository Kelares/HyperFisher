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
# HyperNetwork Bottleneck Experiment (Split-MNIST Single Head)
# ==============================================================================
# Architecture : Extremely Small HyperNetwork (Bottlenecking forced)
# Goal         : Showcase projection methods over-constraining the network.
#                Adam serves as the unconstrained baseline.
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 7 — Split-MNIST SH Suffocated HN, d_h=8  (Sub-RQ1 Panel b B1)
# ALREADY DONE with 5 seeds. Skipping.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 7: SKIPPED — already complete (5 seeds) ==="

TASK="split_mnist_sh"
DEVICE="gpu"

# Hypernetwork specific dimensions
HYPER_HIDDEN=4
TASK_EMBED=32
CHUNK_EMBED=32
CHUNK_SIZE=64

# Training hyperparameters from perfect run
SEEDS=(1) #42 1234 2137 811 111 ) 
EPOCHS=15
BATCH=64
GRADS=100
MAX_DIRS=5000
FISHER=1024

# ── Universal Learning Rate ──────────────────────────────────────────────────
declare -A LR
LR["adam"]="1e-3"
LR["sgd"]="1e-3"
LR["ewc"]="1e-3"
LR["fng"]="1e-3"
LR["ogd"]="1e-3"
LR["ong"]="1e-3"
LR["fopng"]="1e-3"
LR["efopng"]="1e-3"

# ── Per-method lambda (0 for unregularized baselines) ────────────────────────
declare -A LAM
LAM["adam"]="0"
LAM["sgd"]="0"
LAM["ewc"]="400"
LAM["fng"]="1e-3"
LAM["ogd"]="0"
LAM["ong"]="0"
LAM["fopng"]="1e-3"
LAM["efopng"]="1e-3"

ALL_METHODS=("adam" "sgd" "ewc" "fng" "ogd" "ong" "fopng" "efopng")

QUESTION='Primary Sub-RQ1 Panel (b) result on CIFAR-10. Also Sub-
RQ2 Condition 3 (full normalization as proposed) and
Sub-RQ3 Condition B (AdamW first task).'


echo "|----------SOLVES-------------|: ${Question}"
echo "======================================================================"
echo " Starting HyperNetwork Bottleneck Run on $TASK"
echo " Batch=$BATCH | Epochs=$EPOCHS | Fisher Samples=$FISHER"
echo " Hyper Params: hidden=$HYPER_HIDDEN, task_emb=$TASK_EMBED, chunk_emb=$CHUNK_EMBED"
echo "======================================================================"

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--> Running method=$METHOD | lr=${LR[$METHOD]} | lam=${LAM[$METHOD]} | seed=$SEED"

        ARGS=(
            --task=$TASK
            --hyper_hidden_dim=$HYPER_HIDDEN
            --task_embedding_dim=$TASK_EMBED
            --chunk_embedding_dim=$CHUNK_EMBED
            --chunk_size=$CHUNK_SIZE
            --regulizer
            --normalize
            --device_mode=$DEVICE
            --max_directions=$MAX_DIRS
            --methods=$METHOD
            --lr=${LR[$METHOD]}
            --grads_per_task=$GRADS
            --max_epochs=$EPOCHS
            --seed=$SEED
            --lam=${LAM[$METHOD]}
            --fisher_samples=$FISHER
            --batch_size=$BATCH
            --experiment_id=407
        )

        # Execute the python script
        python main.py "${ARGS[@]}"

        echo "--> Done: $METHOD seed=$SEED"
    done
done

echo ""
echo "======================================================================"
echo " ALL RUNS COMPLETE"
echo "======================================================================"