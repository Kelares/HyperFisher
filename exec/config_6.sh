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
ALL_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
LEFT_METHODS=("ewc")

SEEDS_3=(111 2137)
SEEDS_5=(42 1234 2137 811 111)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 6 — Split-CIFAR100 MH Standalone  (Sub-RQ4, Sub-RQ1 Panel a B3)
# Table 1: batch=10, epochs=10, first_task=SGD 1e-2, Fisher=1024
# ──────────────────────────────────────────────────────────────────────────────
QUESTION='Does eFOPNG scale to 10 tasks and 10-class per-task splits?
Serves as Panel (a) reference for Sub-RQ1 Benchmark 3.'

echo "|----------SOLVES-------------|: ${Question}"
echo "=== CONFIG 6: Split-CIFAR100 MH Standalone ==="

declare -A LR6
LR6["adam"]="1e-3"; LR6["sgd"]="1e-2"; LR6["ewc"]="1e-2"
LR6["fng"]="1e-2";  LR6["ogd"]="1e-2"; LR6["ong"]="1e-2"
LR6["fopng"]="1e-2"; LR6["efopng"]="1e-2"

declare -A LAM6
LAM6["adam"]="0"; LAM6["sgd"]="0"; LAM6["ewc"]="10"
LAM6["fng"]="1e-3"; LAM6["ogd"]="0"; LAM6["ong"]="0"
LAM6["fopng"]="1e-3"; LAM6["efopng"]="1e-3"

for METHOD in "${LEFT_METHODS[@]}"; do
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_cifar100 --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=80 --max_directions=800
            --fisher_samples=1024
            --lr=${LR6[$METHOD]} --max_epochs=10 --batch_size=10
            --first_task_opt=sgd --first_task_lr=1e-2
            --device_mode=$DEVICE --seed=$SEED --experiment_id=406
        )
        [ "${LAM6[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM6[$METHOD]})
        echo "--> C6 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done
