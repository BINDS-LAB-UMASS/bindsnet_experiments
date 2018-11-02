#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/dqn_determ_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
time=${2:-50}
n_episodes=${3:-100}
n_snn_episodes=${4:-100}
percentile=${5:-99}

cd ../../../experiments/conversion/
source activate py36

echo $seed $time $n_episodes $n_snn_episodes $percentile

python dqn_determ.py --train --seed $seed --time $time --n_episodes $n_episodes \
                     --n_snn_episodes $n_snn_episodes --percentile $percentile
python dqn_determ.py --test --seed $seed --time $time --n_episodes $n_episodes \
                     --n_snn_episodes $n_snn_episodes --percentile $percentile

exit
