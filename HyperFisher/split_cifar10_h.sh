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
python main.py --task=split_cifar10 --chunk_size=10000 --lam=5e-4 --alpha=0.5 --grads_per_task=76 --max_direction=120 --hyper_hidden_dim=100 --methods fopng --task_embedding_dim=15 --chunk_embedding_dim=15 --epochs=7#   Task 1 Acc: 97.7%
