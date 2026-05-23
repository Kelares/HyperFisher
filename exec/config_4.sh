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
# "efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
DEVICE="gpu"
PROJ_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc")
ALL_METHODS=("ewc")  #"efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
SEEDS_3=(42 1234 811)
SEEDS_5=(42 1234 2137 811 111)

QUESTION='Do projection methods outperform EWC and baselines
on a visual benchmark? Serves as Panel (a) reference for
Sub-RQ1 Benchmark 2'

 
echo "|----------SOLVES-------------|: ${Question}"
echo "=== CONFIG 4: Split-CIFAR10 MH Standalone (Adam first task) ==="

declare -A LR4
LR4["adam"]="1e-3"; LR4["sgd"]="5e-2"; LR4["ewc"]="1e-2"
LR4["fng"]="1e-2";  LR4["ogd"]="5e-2"; LR4["ong"]="1e-2" # FNG AND ONG SHOULD MIMIC EACH OTHER.
LR4["fopng"]="1e-3"; LR4["efopng"]="1e-3"

declare -A LAM4
LAM4["adam"]="0"; LAM4["sgd"]="0"; LAM4["ewc"]="50"
LAM4["fng"]="1e-3"; LAM4["ogd"]="0"; LAM4["ong"]="1e-3"
LAM4["fopng"]="1e-3"; LAM4["efopng"]="1e-3"

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_cifar10 --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=80 --max_directions=400
            --fisher_samples=1024
            --lr=${LR4[$METHOD]} --max_epochs=5 --batch_size=10
            --first_task_opt=adam --first_task_lr=1e-3
            --device_mode=$DEVICE --seed=$SEED --experiment_id=404
        )
        [ "${LAM4[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM4[$METHOD]})
        echo "--> C4 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done