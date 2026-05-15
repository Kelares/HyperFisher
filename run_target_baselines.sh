#!/bin/bash

# ==============================================================================
# Experiment Suite: Target Network (Multi-Head MLP)
# ==============================================================================
# Group: target_network
# Tasks: split_cifar10
# Seeds: 42, 1234, 2137 | NEW 811 111
# Purpose: Establish the physical upper-bound/baselines for CL performance.
# ==============================================================================

TASK="split_cifar10"
SEEDS=(42 1000 2137 811 111)
METHODS=("sgd" "adam" "ogd" "ognd" "fng" "fopng" "prefopng" "efopng")

# Loop through each method defined in the pipeline
for METHOD in "${METHODS[@]}"; do
    # Loop through each seed for statistical Mean/StdDev
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------"
        echo "LAUNCHING: Method=$METHOD | Seed=$SEED"
        echo "ARCH: Target Network (MLP)"
        echo "----------------------------------------------------------"
        
        # Using the TargetNetwork flag as defined in your main.py choices
        python main.py \
            --task=$TASK \
            --model=TargetNetwork \
            --methods=$METHOD \
            --seed=$SEED \
            --device_mode=gpu \
            --lr=1e-2 \
            --batch_size=64 \
            --max_epochs=50 \
            --grads_per_task=250 \
            --max_directions=5000 \
            --fisher_samples=1024 \
            --normalize \
            --no-regulizer \
            --experiment_id=2

        echo "Finished run for $METHOD with seed $SEED"
        echo ""
    done
done

echo "All Target Network experiments completed."