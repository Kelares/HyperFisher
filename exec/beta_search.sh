#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-12:00  # Reduced time limit for grid search
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

# EXPERIMENT_ID=7__ MEANS HYPERNETWORK ABLATION

# ==============================================================================
# HYPERPARAMETER SEARCH SCRIPT
# Objective: Find stable \beta and chunk_size per architecture class. 
# Strategy: Single seed (42), reduced epochs, fast fail.
# ==============================================================================

DEVICE="gpu"
# SEEDS_3=(42 1234 811)
SEED=42
METHOD="adam" # Isolate Hypernetwork mechanics from projection math

# ──────────────────────────────────────────────────────────────────────────────
# SWEEP A: The MNIST Scale (Base: Config 7)
# Target: Small MLPs, suffocated HN. Sweep beta.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== SWEEP A: Split-MNIST SH ==="
BETA_A=(0.01 0.05 0.1)

for BETA in "${BETA_A[@]}"; do
    echo "--> MNIST Sweep: Beta=$BETA"
    python main.py \
        --task=split_mnist_sh --methods=$METHOD --regulizer \
        --normalize --hyper_hidden_dim=8 --task_embedding_dim=4 --chunk_embedding_dim=10 \
        --beta=$BETA --chunk_size=1000 \
        --lr=1e-3 --max_epochs=5 --batch_size=10 \
        --first_task_opt=adamw --first_task_lr=1e-3 \
        --device_mode=$DEVICE --seed=$SEED --experiment_id=701
done

# ──────────────────────────────────────────────────────────────────────────────
# SWEEP B: The CIFAR-10 Scale (Base: Config 8)
# Target: Medium CNNs. Sweep beta and chunk_size.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== SWEEP B: Split-CIFAR10 ==="
BETA_B=(0.05 0.5) #0.05 0.1 0.5)
CHUNK_B=(64) #(500 2000 5000)

for CHUNK in "${CHUNK_B[@]}"; do
    for BETA in "${BETA_B[@]}"; do
        echo "--> CIFAR-10 Sweep: Chunk=$CHUNK, Beta=$BETA"
        python main.py \
            --task=split_cifar10 --methods=$METHOD --regulizer \
            --normalize --hyper_hidden_dim=32 --task_embedding_dim=16 --chunk_embedding_dim=16 \
            --beta=$BETA --chunk_size=$CHUNK \
            --lr=1e-3 --max_epochs=10 --batch_size=64 \
            --first_task_opt=adamw --first_task_lr=1e-3 \
            --device_mode=$DEVICE --seed=$SEED --experiment_id=702
    done
done

# ──────────────────────────────────────────────────────────────────────────────
# SWEEP C: The CIFAR-100 Scale (Base: Config 11)
# Target: Massive target networks. High gradient explosion risk.
# Note: Reduced to 5 tasks for speed. We just need to check if Task 2 learns.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== SWEEP C: Split-CIFAR100 ==="
BETA_C=(0.01 0.05 0.1)  # Kept smaller due to massive chunk penalties

for CHUNK in "${CHUNK_C[@]}"; do
    for BETA in "${BETA_C[@]}"; do
        echo "--> CIFAR-100 Sweep: Chunk=$CHUNK, Beta=$BETA"
        python main.py \
            --task=split_cifar100 --methods=$METHOD --regulizer \
            --normalize --hyper_hidden_dim=128 --task_embedding_dim=64 --chunk_embedding_dim=64 \
            --num_of_tasks=5 \
            --beta=$BETA --chunk_size=6000 \
            --lr=1e-3 --max_epochs=15 --batch_size=64 \
            --first_task_opt=adamw --first_task_lr=1e-3 \
            --device_mode=$DEVICE --seed=$SEED --experiment_id=703
    done
done

echo "=== HYPERPARAMETER SEARCH COMPLETE ==="