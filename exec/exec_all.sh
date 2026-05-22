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
# 3       Split-MNIST SH             Standalone       4        DONE (3 seeds) SH IS TOO POOR.
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

DEVICE="gpu"
PROJ_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc")
ALL_METHODS=("efopng" "fopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
SEEDS_3=(42 1234 811)
SEEDS_5=(42 1234 2137 811 111)


# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIG 1 — Permuted-MNIST Standalone  (Sub-RQ4)
# # Table 1: batch=10, epochs=5, first_task=SGD at method lr, Fisher=full (60K)
# # ──────────────────────────────────────────────────────────────────────────────


# echo "=== CONFIG 1: Permuted-MNIST Standalone ==="

# declare -A LR1
# # These are the learning rates used for Task 2 onwards
# LR1["adam"]="1e-4"; LR1["sgd"]="5e-3"; LR1["ewc"]="1e-4"
# LR1["fng"]="1e-3";  LR1["ogd"]="5e-3"; LR1["ong"]="5e-3"
# LR1["fopng"]="1e-4"; LR1["efopng"]="1e-4"

# declare -A LAM1
# LAM1["adam"]="0"; LAM1["sgd"]="0"; LAM1["ewc"]="10"
# LAM1["fng"]="1e-3"; LAM1["ogd"]="0"; LAM1["ong"]="0"
# LAM1["fopng"]="1e-2"; LAM1["efopng"]="1e-2"

# for METHOD in "${ALL_METHODS[@]}"; do
#     for SEED in "${SEEDS_3[@]}"; do
#         ARGS=(
#             --task=permuted_mnist --model=TargetNetwork
#             --methods=$METHOD --no-regulizer
#             --grads_per_task=80 --max_directions=400
#             --fisher_samples=60000
#             --lr=${LR1[$METHOD]} --max_epochs=5 --batch_size=10
            
#             # THE FIX: Universal SGD initialization at a convergent learning rate
#             --first_task_opt=sgd --first_task_lr=1e-3 
            
#             --device_mode=$DEVICE --seed=$SEED --experiment_id=401
#         )
#         [ "${LAM1[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM1[$METHOD]})
#         echo "--> C1 $METHOD seed=$SEED"
#         python main.py "${ARGS[@]}"
#     done
# done

# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIG 2 — Split-MNIST Multi-Head Standalone  (Sub-RQ4, Sub-RQ1 Panel a B1)
# # Table 1: batch=10, epochs=5, first_task=SGD at method lr, Fisher=full (~12K)
# # ──────────────────────────────────────────────────────────────────────────────
# echo "=== CONFIG 2: Split-MNIST MH Standalone ==="

# declare -A LR2
# LR2["adam"]="1e-5"; LR2["sgd"]="5e-4"; LR2["ewc"]="5e-4"
# LR2["fng"]="1e-3";  LR2["ogd"]="5e-4"; LR2["ong"]="5e-4"
# LR2["fopng"]="1e-5"; LR2["efopng"]="1e-5"

# declare -A LAM2
# LAM2["adam"]="0"; LAM2["sgd"]="0"; LAM2["ewc"]="400"
# LAM2["fng"]="1e-3"; LAM2["ogd"]="0"; LAM2["ong"]="0"
# LAM2["fopng"]="5e-4"; LAM2["efopng"]="5e-4"

# for METHOD in "${ALL_METHODS[@]}"; do
#     for SEED in "${SEEDS_3[@]}"; do
#         ARGS=(
#             --task=split_mnist --model=TargetNetwork
#             --methods=$METHOD --no-regulizer
#             --grads_per_task=80 --max_directions=400
#             --fisher_samples=12000
#             --lr=${LR2[$METHOD]} --max_epochs=5 --batch_size=10
#             --first_task_opt=sgd --first_task_lr=${LR2[$METHOD]}
#             --device_mode=$DEVICE --seed=$SEED --experiment_id=402
#         )
#         [ "${LAM2[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM2[$METHOD]})
#         echo "--> C2 $METHOD seed=$SEED"
#         python main.py "${ARGS[@]}"
#     done
# done

# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIG 3 — Split-MNIST Single-Head Standalone  (Sub-RQ4)
# # Same HPs as MH but task=split_mnist_sh, single 10-output head
# # ──────────────────────────────────────────────────────────────────────────────
# echo "=== CONFIG 3: Split-MNIST SH Standalone ==="

# for METHOD in "${ALL_METHODS[@]}"; do
#     for SEED in "${SEEDS_3[@]}"; do
#         ARGS=(
#             --task=split_mnist_sh --model=TargetNetwork
#             --methods=$METHOD --no-regulizer
#             --grads_per_task=80 --max_directions=400
#             --fisher_samples=12000
#             --lr=${LR2[$METHOD]} --max_epochs=5 --batch_size=10
#             --first_task_opt=sgd --first_task_lr=${LR2[$METHOD]}
#             --device_mode=$DEVICE --seed=$SEED --experiment_id=403
#         )
#         [ "${LAM2[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM2[$METHOD]})
#         echo "--> C3 $METHOD seed=$SEED"
#         python main.py "${ARGS[@]}"
#     done
# done

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 4 — Split-CIFAR10 MH Standalone, AdamW first task
# (Sub-RQ4, Sub-RQ1 Panel a B2, Sub-RQ3 Condition B)
# Table 1: batch=10, epochs=5, first_task=SGD 1e-3, Fisher=1024
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 4: Split-CIFAR10 MH Standalone (AdamW first task) ==="

declare -A LR4
LR4["adam"]="1e-3"; LR4["sgd"]="5e-2"; LR4["ewc"]="1e-2"
LR4["fng"]="1e-2";  LR4["ogd"]="5e-2"; LR4["ong"]="5e-2"
LR4["fopng"]="1e-3"; LR4["efopng"]="1e-3"

declare -A LAM4
LAM4["adam"]="0"; LAM4["sgd"]="0"; LAM4["ewc"]="50"
LAM4["fng"]="1e-3"; LAM4["ogd"]="0"; LAM4["ong"]="0"
LAM4["fopng"]="1e-3"; LAM4["efopng"]="1e-3"

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_cifar10 --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=80 --max_directions=400
            --fisher_samples=1024
            --lr=${LR4[$METHOD]} --max_epochs=5 --batch_size=10
            --first_task_opt=adamw --first_task_lr=1e-3
            --device_mode=$DEVICE --seed=$SEED --experiment_id=404
        )
        [ "${LAM4[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM4[$METHOD]})
        echo "--> C4 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 5 — Split-CIFAR10 MH Standalone, Adam first task
# (Sub-RQ3 Condition A — paired comparison with Config 4)
# Identical to Config 4 except first_task_opt=adam
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 5: Split-CIFAR10 MH Standalone (Adam first task — Sub-RQ3 Cond A) ==="

for METHOD in "${PROJ_METHODS[@]}"; do     # projection methods only — baselines unaffected
    for SEED in "${SEEDS_3[@]}"; do
        ARGS=(
            --task=split_cifar10 --model=TargetNetwork
            --methods=$METHOD --no-regulizer
            --grads_per_task=80 --max_directions=400
            --fisher_samples=1024
            --lr=${LR4[$METHOD]} --max_epochs=5 --batch_size=10
            --first_task_opt=adam --first_task_lr=1e-3
            --device_mode=$DEVICE --seed=$SEED --experiment_id=405
        )
        [ "${LAM4[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM4[$METHOD]})
        echo "--> C5 $METHOD seed=$SEED"
        python main.py "${ARGS[@]}"
    done
done

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 6 — Split-CIFAR100 MH Standalone  (Sub-RQ4, Sub-RQ1 Panel a B3)
# Table 1: batch=10, epochs=10, first_task=SGD 1e-2, Fisher=1024
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 6: Split-CIFAR100 MH Standalone ==="

declare -A LR6
LR6["adam"]="1e-4"; LR6["sgd"]="5e-3"; LR6["ewc"]="1e-2"
LR6["fng"]="5e-3";  LR6["ogd"]="1e-2"; LR6["ong"]="1e-2"
LR6["fopng"]="5e-3"; LR6["efopng"]="5e-3"

declare -A LAM6
LAM6["adam"]="0"; LAM6["sgd"]="0"; LAM6["ewc"]="50"
LAM6["fng"]="1e-3"; LAM6["ogd"]="0"; LAM6["ong"]="0"
LAM6["fopng"]="1e-3"; LAM6["efopng"]="1e-3"

for METHOD in "${ALL_METHODS[@]}"; do
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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 7 — Split-MNIST SH Suffocated HN, d_h=8  (Sub-RQ1 Panel b B1)
# ALREADY DONE with 5 seeds. Skipping.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 7: SKIPPED — already complete (5 seeds) ==="

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 8 — Split-CIFAR10 Standard HN, full normalization
# (Sub-RQ1 Panel b B2, Sub-RQ2 Condition 3)
# ALREADY DONE with 3 seeds. Skipping.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 8: SKIPPED — already complete (3 seeds) ==="

# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIG 9 — Split-CIFAR10 Standard HN, NO normalization
# # (Sub-RQ2 Condition 1 — negative control, expected to crash)
# # eFOPNG only — 3 seeds sufficient to demonstrate the pathology
# # ──────────────────────────────────────────────────────────────────────────────
# echo "=== CONFIG 9: Split-CIFAR10 HN — No normalization (Sub-RQ2 Cond 1) ==="

# for SEED in "${SEEDS_3[@]}"; do
#     echo "--> C9 efopng seed=$SEED"
#     python main.py \
#         --task=split_cifar10 \
#         --methods=efopng \
#         --regulizer \
#         --hyper_hidden_dim=64 \
#         --task_embedding_dim=32 \
#         --chunk_embedding_dim=32 \
#         --chunk_size=256 \
#         --grads_per_task=80 --max_directions=400 \
#         --fisher_samples=1024 \
#         --lr=1e-3 --max_epochs=50 --batch_size=64 \
#         --lam=1e-3 \
#         --first_task_opt=adamw --first_task_lr=1e-3 \
#         --device_mode=$DEVICE --seed=$SEED --experiment_id=409
#         # NOTE: no --normalize flag — this is the negative control
# done

# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIG 10 — Split-CIFAR10 Standard HN, GRADIENT normalization only
# # (Sub-RQ2 Condition 2)
# # ──────────────────────────────────────────────────────────────────────────────
# echo "=== CONFIG 10: Split-CIFAR10 HN — Gradient-only normalization (Sub-RQ2 Cond 2) ==="

# for SEED in "${SEEDS_3[@]}"; do
#     echo "--> C10 efopng seed=$SEED"
#     python main.py \
#         --task=split_cifar10 \
#         --methods=efopng \
#         --regulizer \
#         --hyper_hidden_dim=64 \
#         --task_embedding_dim=32 \
#         --chunk_embedding_dim=32 \
#         --chunk_size=256 \
#         --grads_per_task=80 --max_directions=400 \
#         --fisher_samples=1024 \
#         --lr=1e-3 --max_epochs=50 --batch_size=64 \
#         --lam=1e-3 \
#         --first_task_opt=adamw --first_task_lr=1e-3 \
#         --normalize_gradients_only \
#         --device_mode=$DEVICE --seed=$SEED --experiment_id=410
# done

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 11 — Split-CIFAR100 Standard HN  (Sub-RQ1 Panel b B3)
# ──────────────────────────────────────────────────────────────────────────────
# echo "=== CONFIG 11: Split-CIFAR100 Standard HN (Sub-RQ1 Panel b B3) ==="

# for METHOD in "${ALL_METHODS[@]}"; do
#     for SEED in "${SEEDS_3[@]}"; do
#         ARGS=(
#             --task=split_cifar100
#             --methods=$METHOD
#             --regulizer
#             --normalize
#             --hyper_hidden_dim=128
#             --task_embedding_dim=64
#             --chunk_embedding_dim=64
#             --chunk_size=6000
#             --grads_per_task=80 --max_directions=400
#             --fisher_samples=1024
#             --lr=1e-3 --max_epochs=50 --batch_size=64
#             --first_task_opt=adamw --first_task_lr=1e-3
#             --device_mode=$DEVICE --seed=$SEED  --experiment_id=411
#         )
#         [ "${LAM4[$METHOD]}" != "0" ] && ARGS+=(--lam=${LAM4[$METHOD]})
#         echo "--> C11 $METHOD seed=$SEED"
#         python main.py "${ARGS[@]}"
#     done
# done

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 12 — Preliminary sweep: d_h=4  (Appendix — shows total failure)
# eFOPNG + Adam only, 2 seeds sufficient
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 12: Appendix sweep d_h=4 (total failure baseline) ==="

for METHOD in "efopng" "adam"; do
    for SEED in 42 1234; do
        echo "--> C12 $METHOD seed=$SEED"
        python main.py \
            --task=split_mnist_sh \
            --methods=$METHOD \
            --regulizer \
            --normalize \
            --hyper_hidden_dim=4 \
            --task_embedding_dim=32 \
            --chunk_embedding_dim=32 \
            --chunk_size=64 \
            --grads_per_task=80 --max_directions=400 \
            --fisher_samples=1024 \
            --lr=1e-3 --max_epochs=15 --batch_size=64 \
            --first_task_opt=adamw --first_task_lr=1e-3 \
            --device_mode=$DEVICE --seed=$SEED \
            --lam=1e-3 \
            --experiment_id=412
    done
done

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 13 — Preliminary sweep: d_h=16  (Appendix — shows ceiling effect)
# eFOPNG + Adam only, 2 seeds sufficient
# ──────────────────────────────────────────────────────────────────────────────
echo "=== CONFIG 13: Appendix sweep d_h=16 (ceiling baseline) ==="

for METHOD in "efopng" "adam"; do
    for SEED in 42 1234; do
        echo "--> C13 $METHOD seed=$SEED"
        python main.py \
            --task=split_mnist_sh \
            --methods=$METHOD \
            --regulizer \
            --normalize \
            --hyper_hidden_dim=16 \
            --task_embedding_dim=32 \
            --chunk_embedding_dim=32 \
            --chunk_size=64 \
            --grads_per_task=80 --max_directions=400 \
            --fisher_samples=1024 \
            --lr=1e-3 --max_epochs=15 --batch_size=64 \
            --first_task_opt=adamw --first_task_lr=1e-3 \
            --device_mode=$DEVICE --seed=$SEED \
            --lam=1e-3 --experiment_id=413
    done
done

echo ""
echo "======================================================================"
echo " ALL 13 CONFIGURATIONS COMPLETE"
echo "======================================================================"