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
# CHANGE TO HN
# ==============================================================================
# Full Thesis Experiment Suite — All 13 Configurations
# ==============================================================================
#
# CONFIG  BENCHMARK                  SETTING          SUB-RQ   STATUS
# ──────  ─────────────────────────  ───────────────  ───────  ──────────
# 1       Permuted-MNIST             Standalone       4        TODO
# 2       Split-MNIST MH             Standalone       4, 1a    TODO
# 3       Split-MNIST SH             Standalone       4        TODO
# 4       Split-CIFAR10 MH           Standalone AdamW 4,1a,3B  TODO
# 5       Split-CIFAR10 MH           Standalone Adam  3A       TODO
# 6       Split-CIFAR100 MH          Standalone       4, 1a    TODO
# 7       Split-MNIST SH             Suffocated HN    1b       DONE (5 seeds)
# 8       Split-CIFAR10              Standard HN      1b, 2C3  DONE (3 seeds)
# 9       Split-CIFAR10              HN no-norm       2C1      TODO
# 10      Split-CIFAR10              HN grad-only     2C2      TODO
# 11      Split-CIFAR100             Standard HN      1b       TODO
# 12      Split-MNIST SH (d_h=4)     Prelim sweep     App.     TODO
# 13      Split-MNIST SH (d_h=16)    Prelim sweep     App.     TODO
#
# Hyperparameters for configs 1-6: Garg et al. (2026) Table 1 exactly.
# Hyperparameters for configs 7-13: custom (justified in Methods section).
# ==============================================================================
# "efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
DEVICE="gpu"
PROJ_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc")
ALL_METHODS=("efopng") #  "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
SEEDS_3= (111 2137)
SEEDS_5=(42 1234 2137 811 111)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 5 — Split-CIFAR10 HN Standalone, Adam first task
# (Sub-RQ3 Condition A — paired comparison with Config 8)
# Identical to Config 8 except first_task_opt=adam
# ──────────────────────────────────────────────────────────────────────────────
QUESTION='Does Adam with coupled L2 on the first task contaminate
gradient memory G relative to AdamW? Paired against
Config 8 (AdamW first task).'

echo "|----------SOLVES-------------|: ${Question}"
echo "=== CONFIG 5: Split-CIFAR10 HN Standalone (Adam first task — Sub-RQ3 Cond A) ==="

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_cifar10 
            --model=HyperNetwork 
            --methods=$METHOD 
            
            # Sub-RQ3 Condition B: AdamW first task
            --first_task_opt=adam
            --first_task_lr=1e-3
            
            # Model & Architecture dimensions
            --task_embedding_dim=16 
            --chunk_embedding_dim=16 
            --hyper_hidden_dim=32 
            
            # Updated hyperparameters from Sweep B
            --normalize
            --chunk_size=6000 
            --beta=0.1 
            --regulizer
            
            # Optimization & Fisher settings
            --lr=1e-3 
            --batch_size=32 
            --max_epochs=50 
            --grads_per_task=80 
            --max_directions=1000 
            --fisher_samples=1024 
            
            # Hardware & Tracking
            --device_mode=$DEVICE 
            --seed=$SEED 
            --experiment_id=405
        )
        echo "--> C8 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done
