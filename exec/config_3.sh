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
ALL_METHODS=("ifopng" "ewc" "fopng" "ifopng" "ogd" "ong" "fng" "adam" "sgd")
SEEDS_3=(42) ## 1234 811)
SEEDS_5=(42 1234 2137 811 111)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 3 — Split-MNIST Single-Head Standalone  (Sub-RQ4)
# Same HPs as MH but task=split_mnist_sh, single 10-output head
# ──────────────────────────────────────────────────────────────────────────────

QUESTION='Does the iFOPNG advantage survive the harder single-
head protocol (?) where separate task heads do not sup-
press output-level interference?'

declare -A LR2
LR2["adam"]="1e-5"; LR2["sgd"]="5e-4"; LR2["ewc"]="5e-4"
LR2["fng"]="1e-3";  LR2["ogd"]="5e-4"; LR2["ong"]="5e-4"
LR2["fopng"]="1e-5"; LR2["ifopng"]="1e-5"

declare -A LAM2
LAM2["adam"]="0"; LAM2["sgd"]="0"; LAM2["ewc"]="400"
LAM2["fng"]="1e-3"; LAM2["ogd"]="0"; LAM2["ong"]="0"
LAM2["fopng"]="5e-4"; LAM2["ifopng"]="5e-4"
 
echo "|----------SOLVES-------------|: ${Question}"
echo "=== CONFIG 3: Split-MNIST SH Standalone ==="

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_mnist_sh --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=80 --max_directions=400
            --fisher_samples=12000
            --lr=${LR2[$METHOD]} --max_epochs=5 --batch_size=10
            --first_task_opt=sgd --first_task_lr=1e-3        # CHANGED THE INITIALIZATION AS SGD FAILS FOR SH
            --device_mode=$DEVICE --seed=$SEED --experiment_id=403
        )
        [ "${LAM2[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM2[$METHOD]})
        echo "--> C3 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done