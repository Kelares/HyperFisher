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
DEVICE="gpu"
PROJ_METHODS=("ifopng" "fopng" "ogd" "ong" "fng" "ewc")
SEEDS_3=(42 1234 811)
SEEDS_5=(42 1234 2137 811 111)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 20 — Split-CIFAR10 MH Standalone, SGD first task. ABLATION. VERIFY AT WHAT LR DOES IT FAIL TO CONVERGE.
# ──────────────────────────────────────────────────────────────────────────────
QUESTION='ABLATION. VERIFY AT WHAT LR DOES IT FAIL TO CONVERGE.'

echo "|----------SOLVES-------------|: ${Question}"
echo "=== CONFIG 20: Split-CIFAR10 MH Standalone (Adam first task — Sub-RQ3 Cond A) ==="
SGD_LRS=("5e-2" "1e-2" "5e-3" "1e-3")


for SEED in "${SEEDS_5[@]}"; do
    for LR in "${SGD_LRS[@]}"; do
        ARGS=(
            --task=split_cifar10 --model=TargetNetwork
            --methods=sgd --no-regulizer
            --grads_per_task=80 --max_directions=400
            --fisher_samples=1024
            --lr=$LR --max_epochs=5 --batch_size=10
            --first_task_opt=sgd --first_task_lr=$LR
            --device_mode=$DEVICE --seed=$SEED --experiment_id=420
            --num_of_tasks=1 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        )
        echo "--> C4 sgd seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done