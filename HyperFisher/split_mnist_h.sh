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
#python main.py --task split_mnist --methods ewc fopng adam --embedding_dim=4 --epochs=5 --fisher_samples=4096 --grads_per_task=80 --max_directions=400 # OLD WORKING SPLIT_MNIST BEFORE CHUNKIHNG
python main.py --task split_mnist --alpha=0.5 --lam=1e-2 --methods fopng --hyper_hidden_dim=30 --chunk_embedding_dim=15 --task_embedding_dim=15 --epochs=7 --lr=1e-4 --fisher_samples=1024 --grads_per_task=100 --max_directions=200 --chunk_size=5000 # THE BEST RUN FOR CHUNKING SO FAR
#   Task 1 Acc: 97.7%
#   Task 2 Acc: 92.7%
#   Task 3 Acc: 91.1%
#   Task 4 Acc: 98.7%
#   Task 5 Acc: 96.4%


python main.py --task split_mnist --alpha=0.3 --lr=5e-4 --lam=1e-4 --methods fopng --hyper_hidden_dim=50 --chunk_embedding_dim=15 --task_embedding_dim=15 --epochs=7 --fisher_samples=1024 --grads_per_task=150 --max_directions=300  --chunk_size=5000


python main.py --task split_mnist --alpha=0.3 --lr=1e-3 --lam=1e-3 --methods fopng --hyper_hidden_dim=100 --chunk_embedding_dim=15 --task_embedding_dim=15 --epochs=5 --fisher_samples=1024 --grads_per_task=150 --max_directions=300  --chunk_size=5000
