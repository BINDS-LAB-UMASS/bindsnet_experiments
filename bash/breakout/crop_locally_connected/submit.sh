#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/crop_locally_connected_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
kernel_size=${2:-'25 36'}
stride=${3:-'5 6'}
n_filters=${4:-25}
n_train=${5:-16000}
n_test=${6:-10000}
inhib=${7:-25.0}
time=${8:-350}
theta_plus=${9:-0.05}
theta_decay=${10:-1e-7}
norm=${11:-0.2}

cd ../../../scripts/breakout/
source activate py36

echo $seed $kernel_size $stride $n_filters $n_train $n_test $inhib $time $theta_plus $theta_decay $norm

python crop_locally_connected.py --train --seed $seed --kernel_size $kernel_size --stride $stride \
                                 --n_filters $n_filters --n_train $n_train --n_test $n_test --inhib $inhib \
                                 --time $time --theta_plus $theta_plus --theta_decay $theta_decay --norm $norm
python crop_locally_connected.py --test --seed $seed --kernel_size $kernel_size --stride $stride \
                                 --n_filters $n_filters --n_train $n_train --n_test $n_test --inhib $inhib \
                                 --time $time --theta_plus $theta_plus --theta_decay $theta_decay --norm $norm
exit
