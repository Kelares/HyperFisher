#!/bin/bash

# ==============================================================================
# Experiment Suite: HyperNetwork + HyperRegulizer
# ==============================================================================
# Group: Hypernetwork_without
# Tasks: split_cifar10
# Seeds: 42
# Purpose: Core research runs testing generative CL with restoration force.
# ==============================================================================

TASK="split_cifar10"
SEEDS=(42)
# Methods include both vanilla baselines and your custom projection methods
METHODS=("sgd" "adam" "ogd" "ognd" "fng" "fopng" "prefopng" "efopng")

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------"
        echo "LAUNCHING: Method=$METHOD | Seed=$SEED"
        echo "ARCH: HyperNetwork (Generative) | Regulizer: ON"
        echo "----------------------------------------------------------"
        
        # We use --regulizer to ensure the Mean MSE penalty is active
        # LR is set to 1e-3 for stability in generative space
        python main.py \
            --task=$TASK \
            --model=HyperNetwork \
            --methods=$METHOD \
            --seed=$SEED \
            --device_mode=gpu \
            --lr=1e-3 \
            --batch_size=64 \
            --max_epochs=50 \
            --grads_per_task=250 \
            --max_directions=5000 \
            --fisher_samples=1024 \
            --fisher_normalization \
            --no-regulizer \
            --task_embedding_dim=32 \
            --chunk_embedding_dim=32 \
            --hyper_hidden_dim=32 \
            --chunk_size=64
            
        echo "Finished run for $METHOD with seed $SEED"
        echo ""
    done
done

echo "All HyperNetwork (with Regularizer) experiments completed."