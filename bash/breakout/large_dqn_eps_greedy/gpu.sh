#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --time=04-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=../../output/large_dqn_eps_greedy_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
time=${2:-50}
n_episodes=${3:-100}
n_snn_episodes=${4:-100}
percentile=${5:-99}
epsilon=${6:-0.05}
game=${7}
ann_value=${8}

cd ../../../experiments/conversion/

echo $seed $time $n_episodes $n_snn_episodes $percentile $game $ann

if [ "$ann_value" == 'true' ]; then

    python large_dqn_eps_greedy.py --seed $seed --time $time --n_episodes $n_episodes \
                               --n_snn_episodes $n_snn_episodes --percentile $percentile \
                               --epsilon $epsilon --game $game --ann
else
    python large_dqn_eps_greedy.py --seed $seed --time $time --n_episodes $n_episodes \
                               --n_snn_episodes $n_snn_episodes --percentile $percentile \
                               --epsilon $epsilon
fi
exit
