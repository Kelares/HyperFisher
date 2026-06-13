#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.%N.err
#SBATCH --gres=gpu:1

# ==============================================================================
# CONFIG 11 — Split-CIFAR100 Standard HN — ONG ONLY
# ==============================================================================
# Reason: 10 previously attempted ONG runs produced only 1 valid result
# (seed=2137 at 28.2%). Seeds 42, 811, 1234 all finished at ~10% (degenerate),
# seed=111 has no finished run at all.
#
# ONG is structurally the least stable method (non-PSD denominator u^Tv can
# go negative, causing step-size blowup). On 10-task CIFAR100 HN the gradient
# signals from many chunks amplify this instability. Some seeds may remain
# degenerate — that outcome is itself reportable.
#
# Runs this script: 4 seeds × 3 LRs = 12 runs
# Seeds: 42, 111, 811, 1234  (seed=2137 already valid at 28.2% — skip)
# LRs:   0.005, 0.001, 0.0005
# ==============================================================================

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

cd ~/HyperFisher/
conda activate venv

DEVICE="gpu"
METHOD="ong"
# seed=2137 already has a valid run at 28.2% — intentionally excluded here
SEEDS=(42 111 811 1234)
LRS=(0.005 0.001 0.0005)

echo "=== CONFIG 11: Split-CIFAR100 Standard HN — ONG (4 seeds × 3 LRs) ==="
echo "    (seed=2137 already valid at 28.2%, not rerun)"

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
        # Note: no --lam for ONG (lam=0)
    done
done

echo "=== ONG done. Expected W&B entries: 12 ==="