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

# 1. Define your sweep ranges (Keep them small at first!)
LR_VALUES=(1e-4 1e-3)
LAM_VALUES=(1e-6 1e-3)
ALPHA_VALUES=(0.1 0.5)
FISHER_SAMPLES_VALUES=(512 1024)
GRADS_PER_TASK_VALUES=(40 80)
HYPER_HIDDEN_DIM_VALUES=(16 32)

echo "Starting FOPNG Grid Sweep..."

# 2. Iterate through every combination
for lr in "${LR_VALUES[@]}"; do
  for lam in "${LAM_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
      for fisher in "${FISHER_SAMPLES_VALUES[@]}"; do
        for grads in "${GRADS_PER_TASK_VALUES[@]}"; do
          for hidden_dim in "${HYPER_HIDDEN_DIM_VALUES[@]}"; do
            
            # Dynamically set max_directions to 2x grads_per_task
            max_dirs=$((grads * 2))
            
            echo ""
            echo "=================================================================="
            echo "RUNNING: lr=$lr | lam=$lam | alpha=$alpha | fisher=$fisher | grads=$grads | max_dirs=$max_dirs | hidden=$hidden_dim"
            echo "=================================================================="
            
            # 3. Launch the experiment
            python main.py \
              --task permuted_mnist \
              --methods fopng \
              --lr $lr \
              --lam $lam \
              --alpha $alpha \
              --fisher_samples $fisher \
              --grads_per_task $grads \
              --max_directions $max_dirs \
              --hyper_hidden_dim $hidden_dim
              
          done
        done
      done
    done
  done
done
