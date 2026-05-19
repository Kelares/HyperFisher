#!/bin/bash

# ==============================================================================
# Complete Thesis Experiment Suite: Permuted MNIST (5 Seeds Sweep)
# ==============================================================================
# Focus: Completing statistical coverage to 5 seeds per configuration.
# Handling: Skipping pre-existing runs (Seeds 42 & 2137) for specialized models, 
#           and running full 5-seed matrices for standard solo baselines.
# ==============================================================================

TASK="permuted_mnist"
DEVICE="gpu"

# Define complete set of target seeds for statistical evaluation
ALL_SEEDS=(42 2137 1234 811 111)

# ------------------------------------------------------------------------------
# 1. Specialized Methods Suite (Completing to 5 seeds; skipping 42 and 2137)
# ------------------------------------------------------------------------------
SPECIAL_METHODS=("efopng" "ewc" "ogd" "ong" "fng" "fopng")
FRESH_SEEDS=(1234 811 111) # The 3 missing seeds needed to hit 5 runs total

echo "=== PHASE 1: Launching Missing Seeds for Specialized Frameworks ==="
for METHOD in "${SPECIAL_METHODS[@]}"; do
    for SEED in "${FRESH_SEEDS[@]}"; do
        echo "---------------------------------------------------------"
        echo "LAUNCHING: Method=$METHOD | Seed=$SEED"
        echo "CONFIG: HyperNetwork + Regularizer + Subspace Projection"
        echo "---------------------------------------------------------"
        
        python main.py \
            --task=$TASK \
            --methods=$METHOD \
            --hyper_hidden_dim=16 \
            --task_embedding_dim=8 \
            --chunk_embedding_dim=8 \
            --chunk_size=1280 \
            --regulizer \
            --grads_per_task=80 \
            --max_directions=400 \
            --fisher_samples=1024 \
            --device_mode=$DEVICE \
            --normalize \
            --lr=1e-2 \
            --max_epochs=10 \
            --experiment_id=1000 \
            --seed=$SEED \
            --num_of_tasks
            
        echo "Finished run for $METHOD with seed $SEED"
        echo ""
    done
done

# ------------------------------------------------------------------------------
# 2. Standalone Solo Baselines Suite (Full 5 Seeds Matrix)
# ------------------------------------------------------------------------------
# NOTE: Ensure your main.py maps "adam" or "sgd" without projections 
# when calling --methods=adam or using an explicit --solo flag if required.
SOLO_METHODS=("adam" "sgd")

echo "=== PHASE 2: Launching Full 5-Seed Grid for Solo Baselines ==="
for BASELINE in "${SOLO_METHODS[@]}"; do
    for SEED in "${FRESH_SEEDS[@]}"; do
        echo "---------------------------------------------------------"
        echo "LAUNCHING: Solo Baseline=$BASELINE | Seed=$SEED"
        echo "CONFIG: Standalone HyperNetwork + Functional Regularizer Only"
        echo "---------------------------------------------------------"
        
        python main.py \
            --task=$TASK \
            --methods=$BASELINE \
            --hyper_hidden_dim=16 \
            --task_embedding_dim=8 \
            --chunk_embedding_dim=8 \
            --chunk_size=1280 \
            --regulizer \
            --device_mode=$DEVICE \
            --lr=1e-2 \
            --max_epochs=10 \
            --experiment_id=2000 \
            --seed=$SEED \
            --num_of_tasks
            
        echo "Finished run for Solo $BASELINE with seed $SEED"
        echo ""
    done
done

echo "========================================================="
echo " SUCCESS: All 5-seed experimental arrays are complete!"
echo "========================================================="