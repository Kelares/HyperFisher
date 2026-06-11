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
# CONFIG 19/20/21 — Permuted-MNIST 20-Task Compression Ablation
# Testing SVD vs FIFO vs STOP when gradient memory overflows
#
# Design: max_directions=400 with grads_per_task=80 forces overflow at task 6
# (5 tasks × 80 = 400 fills the budget; task 6 triggers the first compression).
# All conditions use iFOPNG with MAX accumulation (established as superior in
# Sub-RQ4). Only the compression strategy differs.
#
# NOTE: The SVD condition replicates the default behaviour already present in
# Exp 415 — it is re-run here explicitly for a self-contained comparison.
#
# CONFIG  EXP_ID  COMPRESSION  STATUS
# ──────  ──────  ───────────  ──────
# 19      416     svd          TODO
# 20      417     fifo         TODO
# 21      418     stop         TODO
# ==============================================================================

echo "=== CONFIG 19/20/21: Permuted-MNIST 20 Tasks — Compression Ablation ==="

SEEDS_5=(42 1234 2137 811 111)
DEVICE="gpu"

# Shared hyperparameters — identical to Config 15 (MAX baseline)
METHOD="ifopng"            # MAX accumulation throughout

# max_directions=400 with grads_per_task=80 → overflow starts at task 6
MAX_DIR=400
GRADS_PER_TASK=80

COMPRESSION_TYPES=("fifo" "stop" "svd")

for COMP in "${COMPRESSION_TYPES[@]}"; do

    case $COMP in
        svd)  EXP_ID=424 ;;
        fifo) EXP_ID=425 ;;
        stop) EXP_ID=426 ;;
    esac

    echo ""
    echo "--- Compression: ${COMP^^} (Exp $EXP_ID) ---"

    for SEED in "${SEEDS_5[@]}"; do
        ARGS=(
            --task=split_cifar100
            --model=TargetNetwork
            --methods=$METHOD
            --no-regulizer
            --grads_per_task=200
            --max_directions=400
            --compression=$COMP
            --fisher_samples=60000
            --lr=1e-4
            --max_epochs=5
            --batch_size=10
            --lam=1e-3
            --first_task_opt=sgd
            --first_task_lr=1e-3
            --device_mode=$DEVICE
            --seed=$SEED
            --experiment_id=$EXP_ID
        )

        echo "--> C${EXP_ID} iFOPNG compression=${COMP} seed=$SEED"
        python main.py "${ARGS[@]}"
    done

done

echo ""
echo "=== Compression ablation complete. Experiment IDs: 416 (SVD), 417 (FIFO), 418 (STOP) ==="