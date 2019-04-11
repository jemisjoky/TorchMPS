#!/bin/sh
#SBATCH --gres=gpu:1

# bond_dim = 10
# periodic_bc = 1
# random_path = 1
srun train_script.py --lr 0.0001 --init_std 1e-09 --l2_reg 0.0 --num_train 60000 --batch_size 100 --bond_dim 10 --num_epochs 30 --num_test 10000 --adaptive_mode 0 --periodic_bc 1 --merge_threshold 2000 --cutoff 1e-10 --use_gpu 1 --random_path 1  >> RANDOM_PATH_bd_10_bc_1_path_1.log
echo "Done with experiment 1"

# bond_dim = 20
# periodic_bc = 1
# random_path = 1
srun train_script.py --lr 0.0001 --init_std 1e-09 --l2_reg 0.0 --num_train 60000 --batch_size 100 --bond_dim 20 --num_epochs 30 --num_test 10000 --adaptive_mode 0 --periodic_bc 1 --merge_threshold 2000 --cutoff 1e-10 --use_gpu 1 --random_path 1  >> RANDOM_PATH_bd_20_bc_1_path_1.log
echo "Done with experiment 2"

# bond_dim = 40
# periodic_bc = 1
# random_path = 1
srun train_script.py --lr 0.0001 --init_std 1e-09 --l2_reg 0.0 --num_train 60000 --batch_size 100 --bond_dim 40 --num_epochs 30 --num_test 10000 --adaptive_mode 0 --periodic_bc 1 --merge_threshold 2000 --cutoff 1e-10 --use_gpu 1 --random_path 1  >> RANDOM_PATH_bd_40_bc_1_path_1.log
echo "Done with experiment 3"

# bond_dim = 60
# periodic_bc = 1
# random_path = 1
srun train_script.py --lr 0.0001 --init_std 1e-09 --l2_reg 0.0 --num_train 60000 --batch_size 100 --bond_dim 60 --num_epochs 30 --num_test 10000 --adaptive_mode 0 --periodic_bc 1 --merge_threshold 2000 --cutoff 1e-10 --use_gpu 1 --random_path 1  >> RANDOM_PATH_bd_60_bc_1_path_1.log
echo "Done with experiment 4"

# bond_dim = 80
# periodic_bc = 1
# random_path = 1
srun train_script.py --lr 0.0001 --init_std 1e-09 --l2_reg 0.0 --num_train 60000 --batch_size 100 --bond_dim 80 --num_epochs 30 --num_test 10000 --adaptive_mode 0 --periodic_bc 1 --merge_threshold 2000 --cutoff 1e-10 --use_gpu 1 --random_path 1  >> RANDOM_PATH_bd_80_bc_1_path_1.log
echo "Done with experiment 5"

# bond_dim = 100
# periodic_bc = 1
# random_path = 1
srun train_script.py --lr 0.0001 --init_std 1e-09 --l2_reg 0.0 --num_train 60000 --batch_size 100 --bond_dim 100 --num_epochs 30 --num_test 10000 --adaptive_mode 0 --periodic_bc 1 --merge_threshold 2000 --cutoff 1e-10 --use_gpu 1 --random_path 1  >> RANDOM_PATH_bd_100_bc_1_path_1.log
echo "Done with experiment 6"

echo "Done with all experiments"
