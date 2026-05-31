#!/bin/bash

# ==============================================================================
# Complete Thesis Experiment Suite: Split MNIST (Target Network Baselines)
# ==============================================================================
# Focus: Completing full 5-seed statistical matrices for standalone MLP models.
# Handling: Solo methods (Adam/SGD) are isolated to their optimal stable 1e-3 LR,
#           while geometric/projection methods use their required 1e-2 LR.
# Architecture: MultiHeadMLP (Task-Incremental configuration)
# ==============================================================================

TASK="split_cifar10"
MODEL_TYPE="TargetNetwork"
DEVICE="gpu"
EXP_ID=2

# Full 5-seed array for complete statistical coverage (Mean ± StdDev)
ALL_SEEDS=(42 2137 1234 811 111)
# SPECIAL_SEEDS=(111 ) #42 2137)
# ------------------------------------------------------------------------------
# PHASE 1: Geometric & Projection Methods Suite (LR Locked to 1e-2)
# ------------------------------------------------------------------------------
# These methods require higher velocity to compensate for subspace tracking vector clipping.
GEOMETRIC_METHODS=("ewc") # "ong" "ewc" "fng" "fopng" "prefopng" "ifopng")

echo "=== PHASE 1: Launching Subspace Projections and Geometric CL (LR=1e-2) ==="
for METHOD in "${GEOMETRIC_METHODS[@]}"; do
    for SEED in "${ALL_SEEDS[@]}"; do
        echo "---------------------------------------------------------"
        echo "LAUNCHING: Method=$METHOD | Seed=$SEED"
        echo "CONFIG: Model=$MODEL_TYPE | Task=$TASK | LR=1e-2"
        echo "---------------------------------------------------------"
        
        python main.py \
            --task=$TASK \
            --model=$MODEL_TYPE \
            --methods=$METHOD \
            --seed=$SEED \
            --device_mode=$DEVICE \
            --lr=5e-4 \
            --lam=400 \
            --batch_size=64 \
            --max_epochs=50 \
            --grads_per_task=250 \
            --max_directions=5000 \
            --fisher_samples=1024 \
            --normalize \
            --no-regulizer \
            --experiment_id=$EXP_ID
            
        echo "Finished run for $METHOD with seed $SEED"
        echo ""
    done
done

# ------------------------------------------------------------------------------
# PHASE 2: Standalone Unconstrained Baselines Suite (LR LOCKED TO 1e-3)
# # ------------------------------------------------------------------------------
# # Isolating Adam and SGD to their true stable operating regime to prevent artificial decay.
# SOLO_METHODS=("adam" "sgd")

# echo "=== PHASE 2: Launching Tuned Corridors for Solo Baselines (LR=1e-3) ==="
# for BASELINE in "${SOLO_METHODS[@]}"; do
#     for SEED in "${ALL_SEEDS[@]}"; do
#         CURRENT_LR="1e-3"
        
#         echo "---------------------------------------------------------"
#         echo "LAUNCHING: Solo Baseline=$BASELINE | Seed=$SEED"
#         echo "CONFIG: Model=$MODEL_TYPE | Task=$TASK | LR=$CURRENT_LR"
#         echo "---------------------------------------------------------"
        
#         python main.py \
#             --task=$TASK \
#             --model=$MODEL_TYPE \
#             --methods=$BASELINE \
#             --seed=$SEED \
#             --device_mode=$DEVICE \
#             --lr=$CURRENT_LR \
#             --batch_size=64 \
#             --max_epochs=50 \
#             --grads_per_task=250 \
#             --max_directions=5000 \
#             --fisher_samples=1024 \
#             --normalize \
#             --no-regulizer \
#             --experiment_id=$EXP_ID
            
#         echo "Finished run for Solo $BASELINE with seed $SEED"
#         echo ""
#     done
# done

# echo "========================================================="
# echo " SUCCESS: All Split MNIST target matrix runs are complete!"
# echo "========================================================="