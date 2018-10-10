#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/two_level_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_neurons=${2:-100}
n_train=${3:-5500}
n_test=${4:-1300}
start_inhib=${5:-1.0}
max_inhib=${6:-100.0}
p_low=${7:-0.1}
norm=${8:-65}
time=${9:-350}
theta_plus=${10:-0.05}
theta_decay=${11:-1e-7}

cd ../../../experiments/breakout/
source activate py36

echo $seed $n_neurons $n_train $n_test $start_inhib $max_inhib $p_low $norm $time $theta_plus $theta_decay

python two_level.py --train --seed $seed --n_neurons $n_neurons --n_train $n_train --n_test $n_test \
                    --start_inhib $start_inhib --max_inhib $max_inhib --p_low $p_low --norm $norm \
		            --time $time --theta_plus $theta_plus --theta_decay $theta_decay
python two_level.py --test --seed $seed --n_neurons $n_neurons --n_train $n_train --n_test $n_test \
                    --start_inhib $start_inhib --max_inhib $max_inhib --p_low $p_low --norm $norm \
                    --time $time --theta_plus $theta_plus --theta_decay $theta_decay
exit
