#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/unclamp_dac_mnist_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_neurons=${2:-100}
n_train=${3:-60000}
n_test=${4:-10000}
inhib=${5:-25.0}
time=${6:-350}
theta_plus=${7:-0.05}
theta_decay=${8:-1e-7}
intensity=${9:-0.5}
lr=${10:-0.01}
lr_decay=${11:0.99}

cd ../../../experiments/mnist/
source activate py36

echo $seed $n_neurons $n_train $n_test $inhib $time $theta_plus $theta_decay $intensity $lr $lr_decay

python unclamp_dac.py --train --seed $seed --n_neurons $n_neurons --n_train $n_train \
					  --n_test $n_test --inhib $inhib --time $time \
					  --theta_plus $theta_plus --theta_decay $theta_decay \
					  --intensity $intensity --lr $lr --lr_decay $lr_decay
python unclamp_dac.py --test --seed $seed --n_neurons $n_neurons --n_train $n_train \
					  --n_test $n_test --inhib $inhib --time $time \
					  --theta_plus $theta_plus --theta_decay $theta_decay \
					  --intensity $intensity --lr $lr --lr_decay $lr_decay
exit
