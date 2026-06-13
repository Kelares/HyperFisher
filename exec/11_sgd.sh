#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.%N.err
#SBATCH --gres=gpu:1

# ==============================================================================
# CONFIG 11 — Split-CIFAR100 Standard HN — SGD ONLY
# ==============================================================================
# Reason: all 3 previously completed SGD runs landed below the 0.25 validity
# threshold (0.158, 0.179, 0.210). SGD catastrophically forgets on 10-task
# CIFAR100 HN without any memory mechanism — this IS the expected scientific
# result, but we need 5 seeds to report it credibly as a baseline.
#
# Runs this script: 5 seeds × 3 LRs = 15 runs
# Seeds: 42, 111, 811, 1234, 2137 (full preferred set)
# LRs:   0.005, 0.001, 0.0005  (same sweep as original config)
# ==============================================================================

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

cd ~/HyperFisher/
conda activate venv

DEVICE="gpu"
METHOD="sgd"
SEEDS=(42 111 811 1234 2137)
LRS=(0.005 0.001 0.0005)

echo "=== CONFIG 11: Split-CIFAR100 Standard HN — SGD (5 seeds × 3 LRs) ==="

for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "--> C11 $METHOD seed=$SEED lr=$LR"
        python main.py \
            --task=split_cifar100 \
            --methods=$METHOD \
            --regulizer \
            --normalize \
            --hyper_hidden_dim=32 \
            --task_embedding_dim=16 \
            --chunk_embedding_dim=16 \
            --chunk_size=6000 \
            --grads_per_task=200 --max_directions=800 \
            --fisher_samples=1024 \
            --beta=0.1 \
            --lr=$LR --max_epochs=50 --batch_size=64 \
            --first_task_opt=adamw --first_task_lr=1e-3 \
            --device_mode=$DEVICE --seed=$SEED --experiment_id=411
        # Note: no --lam for SGD (regulariser weight = 0)
    done
done

echo "=== SGD done. Expected W&B entries: 15 ==="