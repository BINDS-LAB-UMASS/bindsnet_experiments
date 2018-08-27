#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/noisy_dac_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_neurons=${2:-100}
n_train=${3:-60000}
n_test=${4:-10000}
inhib=${5:-25.0}
time=${6:-350}
theta_plus=${7:-0.05}
theta_decay=${8:-1e-7}
p=${9:-0.01}
intensity=${10:-0.5}
X_Ae_decay=${11:-0.5}

cd ../../../scripts/mnist/
source activate py36

echo $seed $n_neurons $n_train $n_test $excite $inhib $time $theta_plus $theta_decay $p $intensity $X_Ae_decay

python noisy_dac.py --train --seed $seed --n_neurons $n_neurons --n_train $n_train \
					--n_test $n_test --inhib $inhib --time $time --theta_plus $theta_plus \
					--theta_decay $theta_decay --p $p --intensity $intensity
python noisy_dac.py --test --seed $seed --n_neurons $n_neurons --n_train $n_train \
					--n_test $n_test --inhib $inhib --time $time --theta_plus $theta_plus \
					--theta_decay $theta_decay --p $p --intensity $intensity
exit