#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../../output/crop_locally_connected_neurons_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
kernel_size=${2:-16}
stride=${3:-4}
n_filters=${4:-16}
crop=${5:-4}
n_train=${6:-60000}
n_test=${7:-10000}
inhib=${8:-250.0}
time=${9:-350}
theta_plus=${10:-0.05}
theta_decay=${11:-1e-7}
intensity=${12:-0.5}
norm=${13:-0.2}
lr=${14:-0.01}
lr_decay=${15:-1}
p_remove=${16:-0.1}
new_seed=${17:-0}

cd ../../../../experiments/robustness/mnist/
source activate py36

echo $seed $kernel_size $stride $n_filters $crop $n_train $n_test $inhib $time $theta_plus $theta_decay $intensity \
     $norm $lr $lr_decay $p_remove $new_seed

python crop_locally_connected_neurons.py --seed $seed --kernel_size $kernel_size --stride $stride \
                                         --n_filters $n_filters --crop $crop --n_train $n_train \
                                         --inhib $inhib --time $time --theta_plus $theta_plus \
                                         --theta_decay $theta_decay --intensity $intensity --norm $norm \
                                         --lr $lr --lr_decay $lr_decay --p_remove $p_remove --new_seed $new_seed
exit
