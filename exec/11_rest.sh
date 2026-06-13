#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.%N.err
#SBATCH --gres=gpu:1

# ==============================================================================
# CONFIG 11 — Split-CIFAR100 Standard HN — REMAINING METHODS (targeted gaps)
# ==============================================================================
# EWC and iFOPNG already have 5/5 valid seeds — NOT rerun here.
#
# Method   Missing seeds   Previous best     Reason
# ───────  ─────────────   ─────────────     ──────────────────────────────────
# adam     811             0.141 (degen.)    Degenerate; all other 4 seeds ok
# ogd      111, 2137       0.211, 0.233      Below 0.25 threshold; borderline
# fopng    111             0.237             Below 0.25 threshold; borderline
# fng      1234            0.241             Below 0.25 threshold; borderline
#
# Runs this script: 3 + 6 + 3 + 3 = 15 runs
# LRs: 0.005, 0.001, 0.0005 (same sweep as all other config-11 runs)
# ==============================================================================

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

cd ~/HyperFisher/
conda activate venv

DEVICE="gpu"
LRS=(0.005 0.001 0.0005)

# Lambda values (0 = flag not passed)
declare -A LAM
LAM["adam"]="0"
LAM["ogd"]="0"
LAM["fopng"]="1e-3"
LAM["fng"]="1e-3"

# Targeted seed lists — only the seeds that are missing or degenerate
declare -A SEEDS_NEEDED
SEEDS_NEEDED["adam"]="811"
SEEDS_NEEDED["ogd"]="111 2137"
SEEDS_NEEDED["fopng"]="111"
SEEDS_NEEDED["fng"]="1234"

echo "=== CONFIG 11: Split-CIFAR100 Standard HN — Targeted gap-fill ==="
echo "    adam seed=811 | ogd seeds=111,2137 | fopng seed=111 | fng seed=1234"
echo "    EWC and iFOPNG: already 5/5 valid — skipped"

for METHOD in "adam" "ogd" "fopng" "fng"; do
    read -ra SEEDS <<< "${SEEDS_NEEDED[$METHOD]}"
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LRS[@]}"; do
            echo "--> C11 $METHOD seed=$SEED lr=$LR"
            ARGS=(
                --task=split_cifar100
                --methods=$METHOD
                --regulizer
                --normalize
                --hyper_hidden_dim=32
                --task_embedding_dim=16
                --chunk_embedding_dim=16
                --chunk_size=6000
                --grads_per_task=200 --max_directions=800
                --fisher_samples=1024
                --beta=0.1
                --lr=$LR --max_epochs=50 --batch_size=64
                --first_task_opt=adamw --first_task_lr=1e-3
                --device_mode=$DEVICE --seed=$SEED --experiment_id=411
            )
            [ "${LAM[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM[$METHOD]})
            python main.py "${ARGS[@]}"
        done
    done
done

echo "=== Gap-fill done. Expected W&B entries: 15 ==="