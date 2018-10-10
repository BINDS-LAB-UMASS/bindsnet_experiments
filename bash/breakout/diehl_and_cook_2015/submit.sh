#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/diehl_and_cook_2015_breakout_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_neurons=${2:-100}
n_train=${3:-6750}
n_test=${4:-6750}
inhib=${5:-25.0}
time=${6:-350}
theta_plus=${7:-0.05}
theta_decay=${8:-1e-7}

cd ../../../experiments/breakout/
source activate py36

echo $seed $n_neurons $n_train $n_test $inhib $time $theta_plus $theta_decay

python diehl_and_cook_2015.py --train --seed $seed --n_neurons $n_neurons --n_train $n_train \
							  --n_test $n_test --inhib $inhib --time $time \
							  --theta_plus $theta_plus --theta_decay $theta_decay
python diehl_and_cook_2015.py --test --seed $seed --n_neurons $n_neurons --n_train $n_train \
							  --n_test $n_test --inhib $inhib --time $time \
							  --theta_plus $theta_plus --theta_decay $theta_decay
exit
