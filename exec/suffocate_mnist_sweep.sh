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
ALL_METHODS=("efopng" "fopng" "efopng" "ogd" "ong" "fng" "ewc" "adam" "sgd")
SEEDS_3=(42 1234 811)
SEEDS_5=(42 1234 2137 811 111)
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