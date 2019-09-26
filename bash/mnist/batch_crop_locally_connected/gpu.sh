#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=00-04:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=../../output/batch_crop_locally_connected_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_epochs=${2:-1}
batch_size=${3:-32}
inhib=${4:-250.0}
kernel_size=${5:-16}
stride=${6:-2}
n_filters=${7:-25}
crop=${8:-0}
lr=${9:-0.01}
lr_decay=${10:-1}
time=${11:-100}
theta_plus=${12:-0.05}
theta_decay=${13:-1e-7}
intensity=${14:-5}
norm=${13:-0.2}

cd ../../../experiments/mnist/
source activate py36

echo $seed $n_epochs $batch_size $inhib $kernel_size $stride $n_filters $crop $lr $lr_decay $time $theta_plus \
     $theta_decay $intensity $norm

python batch_crop_locally_connected.py --train --gpu --seed $seed --n_epochs $n_epochs --batch_size $batch_size\
                                 --inhib $inhib --kernel_size $kernel_size --stride $stride --n_filters $n_filters\
                                  --crop $crop --lr $lr --lr_decay $lr_decay --time $time --theta_plus $theta_plus \
                                 --theta_decay $theta_decay --intensity $intensity --norm $norm
exit
