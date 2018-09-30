#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=../../output/bern_crop_locally_connected_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
kernel_size=${2:-16}
stride=${3:-4}
n_filters=${4:-16}
crop=${5:-4}
max_prob=${6:-0.5}
n_train=${7:-60000}
n_test=${8:-10000}
inhib=${9:-250.0}
time=${10:-350}
theta_plus=${11:-0.05}
theta_decay=${12:-1e-7}
intensity=${13:-0.5}
norm=${14:-0.2}

cd ../../../scripts/mnist/
source activate py36

echo $seed $kernel_size $stride $n_filters $crop $max_prob $n_train \
     $n_test $inhib $time $theta_plus $theta_decay $intensity $norm

python bern_crop_locally_connected.py --train --seed $seed --kernel_size $kernel_size --stride $stride \
                                      --n_filters $n_filters --crop $crop --max_prob $max_prob --n_train $n_train \
                                      --n_test $n_test --inhib $inhib --time $time --theta_plus $theta_plus \
                                      --theta_decay $theta_decay --intensity $intensity --norm $norm
python bern_crop_locally_connected.py --test --seed $seed --kernel_size $kernel_size --stride $stride \
                                      --n_filters $n_filters --crop $crop --max_prob $max_prob --n_train $n_train \
                                      --n_test $n_test --inhib $inhib --time $time --theta_plus $theta_plus \
                                      --theta_decay $theta_decay --intensity $intensity --norm $norm
exit
