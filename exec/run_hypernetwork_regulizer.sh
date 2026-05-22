#!/bin/bash

# ==============================================================================
# Complete Thesis Experiment Suite: Locked Optimal HyperNet Runs
# ==============================================================================
# Focus: Data-driven completion of the 5-seed grid for HyperNet_Reg_True.
# Rule: No overlapping sweeps. Executes only top-performing corridors.
# ==============================================================================

TASK="split_cifar10"
DEVICE="gpu"

# Common structural arguments for all chunked hypernetwork tracks
HNET_ARGS="--task=$TASK --hyper_hidden_dim=32 --task_embedding_dim=32 --chunk_embedding_dim=32 --chunk_size=64 --regulizer --device_mode=$DEVICE --max_directions=5000"

echo "=== PHASE 1: Completing Core Subspace Projections (LR=1e-2, Grads=250) ==="
CORE_PROJS=("ewc" ) #"fopng" "fng" "ogd")
for METHOD in "${CORE_PROJS[@]}"; do
    for SEED in 811 ; do
        echo "Running: $METHOD | Seed: $SEED"
        python main.py $HNET_ARGS --methods=$METHOD --lr=1e-2 --lam=100 --grads_per_task=250 --normalize --max_epochs=50 --seed=$SEED
    done
done

# echo "=== PHASE 2: Completing ONG Subspace Divergence (LR=1e-2, Grads=250) ==="
# for SEED in 111; do
#     echo "Running: ONG | Seed: $SEED"
#     python main.py $HNET_ARGS --methods=ong --lr=1e-2 --grads_per_task=250 --normalize --max_epochs=50 --seed=$SEED
# done

# echo "=== PHASE 3: Completing EWC Regularization (LR=1e-2, Grads=40) ==="
# for SEED in 42 1234 2137 811 111; do
#     echo "Running: EWC | Seed: $SEED"
#     python main.py $HNET_ARGS --methods=ewc --lr=1e-2 --grads_per_task=250 --max_epochs=50 --seed=$SEED
# done

# echo "=== PHASE 4: Completing Tuned Adam Baselines (LR=1e-3, Grads=40) ==="
# for SEED in 2137; do
#     echo "Running: Adam Solo | Seed: $SEED"
#     python main.py $HNET_ARGS --methods=adam --lr=1e-3 --max_epochs=50 --seed=$SEED
# done

# echo "=== PHASE 5: Compiling Tuned SGD Baselines (LR=1e-2, Grads=250) ==="
# for SEED in 811 111; do
#     echo "Running: SGD Solo | Seed: $SEED"
#     python main.py $HNET_ARGS --methods=sgd --lr=1e-3 --max_epochs=50 --seed=$SEED
# done

echo "========================================================="
echo " SUCCESS: Targeted optimal matrix has been successfully dispatched!"
echo "========================================================="