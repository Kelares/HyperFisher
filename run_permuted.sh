#!/bin/bash
# ==============================================================================
# Thesis Experiment Suite — Permuted-MNIST
# ==============================================================================
# Two panels (mirrors the Split-CIFAR10 figure structure):
#   Panel (a): TargetNetwork (standalone) — projection methods beat Adam
#   Panel (b): HyperNetwork              — Adam beats projection methods
#
# 3 seeds × 8 methods × 2 panels = 48 runs
# Estimated time per run: ~8 min  →  total ~6.5h
#
# Speed knobs (edit here):
#   NUM_TASKS    — reduce to 5 for a fast sanity check, 10 for thesis
#   EPOCHS       — 5 is enough for MNIST-scale tasks
#   FISHER_SAMPS — 512 halves Fisher estimation cost vs 1024
#   BATCH        — 256 gives ~4x fewer steps/epoch vs 64
# ==============================================================================

TASK="permuted_mnist"
DEVICE="gpu"
SEEDS=(1234 811)
NUM_TASKS=5
EPOCHS=5
FISHER_SAMPS=512
BATCH=256
LR=1e-3
GRADS=80
MAX_DIRS=400

PROJ_METHODS=("ewc") #, "efopng" "fopng" "ogd" "ong" "fng" "ewc")
BASELINE_METHODS=("adam" "sgd")
ALL_METHODS=("${PROJ_METHODS[@]}" "${BASELINE_METHODS[@]}")

# ==============================================================================
# PANEL (a): Standalone TargetNetwork — no hypernetwork, no functional reg.
# Projection methods should clearly outperform Adam here.
# # ==============================================================================
# echo ""
# echo "======================================================================"
# echo " PANEL (a): TargetNetwork — Standalone Projection vs. Baselines"
# echo "======================================================================"

# for METHOD in "${ALL_METHODS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         echo ""
#         echo "  [TargetNetwork] method=$METHOD  seed=$SEED"

#         python main.py \
#             --task=$TASK \
#             --model=TargetNetwork \
#             --methods=$METHOD \
#             --no-regulizer \
#             --grads_per_task=$GRADS \
#             --max_directions=$MAX_DIRS \
#             --fisher_samples=$FISHER_SAMPS \
#             --device_mode=$DEVICE \
#             --lr=$LR \
#             --max_epochs=$EPOCHS \
#             --batch_size=$BATCH \
#             --num_of_tasks=$NUM_TASKS \
#             --seed=$SEED \
#             --experiment_id=7

#         echo "  Done: TargetNetwork $METHOD seed=$SEED"
#     done
# done

# echo ""
# echo "  Panel (a) complete."
# ==============================================================================
# PANEL (a): Standalone TargetNetwork — no hypernetwork, no functional reg.
# Projection methods should clearly outperform Adam here.
# ==============================================================================
echo ""
echo "======================================================================"
echo " PANEL (a): TargetNetwork — Standalone Projection vs. Baselines"
echo "======================================================================"

for METHOD in "ewc"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "  [TargetNetwork] method=$METHOD  seed=$SEED"

        python main.py \
            --task=$TASK \
            --model=TargetNetwork \
            --methods=$METHOD \
            --no-regulizer \
            --grads_per_task=$GRADS \
            --max_directions=$MAX_DIRS \
            --fisher_samples=$FISHER_SAMPS \
            --device_mode=$DEVICE \
            --lr=$LR \
            --max_epochs=$EPOCHS \
            --batch_size=$BATCH \
            --num_of_tasks=$NUM_TASKS \
            --lam=10 \
            --seed=$SEED \
            --experiment_id=7

        echo "  Done: TargetNetwork $METHOD seed=$SEED"
    done
done

echo ""
echo "  Panel (a) complete."

# ==============================================================================
# PANEL (b): HyperNetwork — functional regularizer, small meta-generator.
# Adam should dominate; projection methods should still beat plain SGD.
#
# Target network: MLP 784→100→100→10  (~90K params)
# Chunks at size 1280: ~71 chunks  (fast, no gradient pathology)
# Hypernetwork: Linear(16→16→1280)  (~24K params)
# ==============================================================================
echo ""
echo "======================================================================"
echo " PANEL (b): HyperNetwork — Functional Regularizer + Projection"
echo "======================================================================"

for METHOD in "${ALL_METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "  [HyperNetwork] method=$METHOD  seed=$SEED"

        python main.py \
            --task=$TASK \
            --methods=$METHOD \
            --regulizer \
            --hyper_hidden_dim=16 \
            --task_embedding_dim=8 \
            --chunk_embedding_dim=8 \
            --chunk_size=1280 \
            --grads_per_task=$GRADS \
            --max_directions=$MAX_DIRS \
            --fisher_samples=$FISHER_SAMPS \
            --device_mode=$DEVICE \
            --normalize \
            --lr=$LR \
            --max_epochs=$EPOCHS \
            --batch_size=$BATCH \
            --num_of_tasks=$NUM_TASKS \
            --seed=$SEED \
            --experiment_id=7

        echo "  Done: HyperNetwork $METHOD seed=$SEED"
    done
done

echo ""
echo "  Panel (b) complete."

echo ""
echo "======================================================================"
echo " ALL RUNS COMPLETE"
echo " Results logged to Weights & Biases."
echo " experiment_id=100 → Panel (a) TargetNetwork"
echo " experiment_id=200 → Panel (b) HyperNetwork"
echo "======================================================================"