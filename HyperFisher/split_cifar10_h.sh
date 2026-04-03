#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm/%j.out # STDOUT
#SBATCH -e slurm/%j.%N.err # STDERR
#SBATCH --gres=gpu:1
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi
cd ~/SSM_benchmark/HyperFisher/
conda activate venv


#if you want to run with single gpus, use the following command

python main.py --task=split_cifar10   --chunk_size=1500   --hyper_hidden_dim=400   --lr=1e-3   --epochs=30   --lam=1e-4   --alpha=0.3   --grads_per_task=500   --max_directions=1000   --methods fopng   --task_embedding_dim=32   --chunk_embedding_dim=32

# python main.py --task=split_cifar10 \
#   --chunk_size=1500 \
#   --hyper_hidden_dim=400 \
#   --lr=1e-4 \
#   --epochs=20 \
#   --lam=1e-2 \
#   --alpha=0.4 \
#   --grads_per_task=250 \
#   --max_directions=1250 \
#   --methods fopng \
#   --task_embedding_dim=32 \
#   --chunk_embedding_dim=32