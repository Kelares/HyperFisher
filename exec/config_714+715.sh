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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 14 & 15 — Permuted-MNIST 20-Task Stress Test (Sub-RQ5)
# Testing EMA vs MAX Fisher Accumulation
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 14 & 15: Permuted-MNIST 20 Tasks (Sub-RQ5) ==="
SEEDS_5=(42 1234 2137 811 111)
DEVICE="gpu"

# For Sub-RQ5, we only need to test iFOPNG
LR="1e-4"
LAM="1e-3"

# Assuming you will add this flag to your argparse in main.py
ACCUMULATION_TYPES=("ema" "max")

for ACC_TYPE in "${ACCUMULATION_TYPES[@]}"; do
    # Assign experiment IDs: 414 for EMA (Config 14), 415 for Max (Config 15)
    if [ "$ACC_TYPE" == "ema" ]; then
        EXP_ID=714
        METHOD="ifopng_ema"
    else
        EXP_ID=715
        METHOD="ifopng"
    fi

    for SEED in "${SEEDS_5[@]}"; do
        ARGS=(
            --task=split_cifar100, --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=200 --max_directions=400
            --fisher_samples=60000
            --lr=$LR 
            --max_epochs=100 --batch_size=10
            --lam=$LAM
            # Universal SGD initialization 
            --first_task_opt=sgd --first_task_lr=1e-3 
            
            --device_mode=$DEVICE --seed=$SEED --experiment_id=$EXP_ID
        )
        
        echo "--> C${EXP_ID} $METHOD (Accumulation: $ACC_TYPE) seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done