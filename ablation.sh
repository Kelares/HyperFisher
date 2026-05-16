
#!/bin/bash

# ==============================================================================
# Experiment Suite: HyperNetwork + HyperRegulizer
# ==============================================================================
# Group: Hypernetwork_with_reg
# Tasks: split_cifar10
# Seeds: 42, 1234, 2137 | 811 111
# Purpose: Core research runs testing generative CL with restoration force.
# ==============================================================================

TASK="permuted_mnist"
SEEDS=(42 2137)
# Methods include both vanilla baselines and your custom projection methods
METHODS=("efopng" "ewc" "ogd" "ong" "fng" "fopng" "prefopng")

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "---------------------------------------------------------"
        echo "LAUNCHING: Method=$METHOD | Seed=$SEED"
        echo "ARCH: HyperNetwork (Generative) | Regulizer: ON"
        echo "----------------------------------------------------------"
        
        # We use --regulizer to ensure the Mean MSE penalty is active
        # LR is set to 1e-3 for stability in generative space

        python main.py \
            --task=permuted_mnist \
            --methods=$METHOD \
            --hyper_hidden_dim=16 \
            --task_embedding_dim=8 \
            --chunk_embedding_dim=8 \
            --chunk_size=1280 \
            --regulizer \
            --grads_per_task=80 \
            --max_directions=400 \
            --fisher_samples=1024 \
            --device_mode=gpu \
            --normalize \
            --lr=1e-2 \
            --max_epochs=10 \
            --experiment_id=1000 \
            --seed=$SEED

            
        echo "Finished run for $METHOD with seed $SEED"
        echo ""
    done
done

echo "All HyperNetwork (with Regularizer) experiments completed."

