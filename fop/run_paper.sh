
optim=fop
# optim=sgd
optim=kfac
batch_size=256
interval=10
lr=0.01 

#if you want to run with single gpus, use the following command
# python train_paper.py --data_augmentation --cutout  --batch_size $batch_size --epochs 100 --curvature_update_interval $interval --beta_adaptive  --eta_adaptive  --optim $optim  --learning_rate $lr

# If you want to run with multiple gpus, use the following command
# gpus=2
# batch_size=50000
# lr=0.1
# torchrun --nproc_per_node=${gpus} --standalone  train_paper.py --data_augmentation --cutout --batch_size $batch_size --epochs 100 --curvature_update_interval $interval --beta_adaptive  --eta_adaptive  --optim $optim  --learning_rate $lr
