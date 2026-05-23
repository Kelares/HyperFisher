#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-04:00
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
# CONFIG 19 — SGD First-Task Convergence Sweep  (Diagnostic)
#
# PURPOSE: Establish the SGD learning rate threshold below which task-1
#          training fails to converge on Split-MNIST SH within 5 epochs.
#          Motivates Adam substitution for all low-lr experiments.
#
# TRACKS:  task-1 accuracy + Fisher max/mean after 5 epochs (via wandb)
# RUNS:    8 LRs × 5 seeds = 40 runs  (~2 min each = ~80 min total)
# ==============================================================================

DEVICE="gpu"
SEEDS=(42 1234 2137 811 111)
SGD_LRS=("1e-2" "5e-3" "1e-3" "5e-4" "2e-4" "1e-4" "5e-5" "1e-5")

echo "=== CONFIG 19: SGD Convergence Threshold Sweep ==="

for LR in "${SGD_LRS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "--> SGD lr=${LR} seed=${SEED}"
        python main.py \
            --task=split_mnist_sh \
            --model=TargetNetwork \
            --methods=sgd \
            --no-regulizer \
            --lr=${LR} \
            --first_task_opt=sgd \
            --first_task_lr=${LR} \
            --max_epochs=5 \
            --batch_size=10 \
            --fisher_samples=12000 \
            --grads_per_task=80 \
            --max_directions=400 \
            --num_of_tasks=1 \
            --device_mode=${DEVICE} \
            --seed=${SEED} \
            --experiment_id=419
    done
done

echo "=== CONFIG 19 complete ==="