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

# wandb agent michalowski-jb-tilburg-university/HyperFisher-HyperFisher_sweeps/iic0k4bb # FOR OLD MNIST
wandb agent michalowski-jb-tilburg-university/HyperFisher-HyperFisher/cvxa4512 # CIFAR