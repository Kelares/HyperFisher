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
# "ifopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
DEVICE="gpu"
PROJ_METHODS=("ifopng" "fopng" "ogd" "ong" "fng" "ewc")
ALL_METHODS=("ifopng") #"fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
SEEDS_3=(42 1234 811)
SEEDS_5=(42 1234 2137 811 111)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 2 — Split-MNIST Multi-Head Standalone  (Sub-RQ4, Sub-RQ1 Panel a B1)
# Table 1: batch=10, epochs=5, first_task=SGD at method lr, Fisher=full (~12K)
# ──────────────────────────────────────────────────────────────────────────────

QUESTION='does iFOPNG replicate and extend FOPNG’s canonical
result? Serves as Panel (a) unconstrained reference for
Sub-RQ1 Benchmark 1'

 
echo "|----------SOLVES-------------|: ${Question}"
echo "=== CONFIG 2: Split-MNIST MH Standalone ==="

declare -A LR2
LR2["adam"]="1e-5"; LR2["sgd"]="5e-4"; LR2["ewc"]="5e-4"
LR2["fng"]="1e-3";  LR2["ogd"]="5e-4"; LR2["ong"]="5e-4"
LR2["fopng"]="1e-5"; LR2["ifopng"]="1e-5"

declare -A LAM2
LAM2["adam"]="0"; LAM2["sgd"]="0"; LAM2["ewc"]="400"
LAM2["fng"]="1e-3"; LAM2["ogd"]="0"; LAM2["ong"]="0"
LAM2["fopng"]="5e-4"; LAM2["ifopng"]="5e-4"

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_mnist_mh --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=80 --max_directions=400
            --fisher_samples=12000
            --lr=${LR2[$METHOD]} --max_epochs=5 --batch_size=10
            # THE FIX: Universal SGD initialization at a convergent learning rate
            --first_task_opt=sgd --first_task_lr=1e-3 
            --device_mode=$DEVICE --seed=$SEED --experiment_id=402
        )
        [ "${LAM2[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM2[$METHOD]})
        echo "--> C2 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done