#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/full_conv_mnist_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
kernel_size=${2:-16}
stride=${3:-4}
n_filters=${4:-16}
n_train=${5:-60000}
n_test=${6:-10000}
inhib=${7:-250.0}
time=${8:-350}
theta_plus=${9:-0.05}
theta_decay=${10:-1e-7}
intensity=${11:-0.5}

cd ../../../scripts/mnist/
source activate py36

echo $seed $kernel_size $stride $n_filters $n_train $n_test $inhib $time $theta_plus $theta_decay $intensity

python fully_conv.py --train --seed $seed --kernel_size $kernel_size --stride $stride \
                     --n_filters $n_filters --n_train $n_train --n_test $n_test \
                     --inhib $inhib --time $time --theta_plus $theta_plus \
                     --theta_decay $theta_decay --intensity $intensity
python fully_conv.py --test --seed $seed --kernel_size $kernel_size --stride $stride \
                     --n_filters $n_filters --n_train $n_train --n_test $n_test \
                     --inhib $inhib --time $time --theta_plus $theta_plus \
                     --theta_decay $theta_decay --intensity $intensity
exit
