#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=../../output/convert_and_save_snn_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
time=${2:-50}
n_episodes=${3:-100}
percentile=${4:-99}
epsilon=${6:-0.05}
game=${7}
normalize_spikes=${8}

cd ../../../experiments/conversion/

echo $seed $time $n_episodes $n_snn_episodes $percentile $game $normalize_spikes

if [ "$normalize_spikes" == 'true' ]; then

    python convert_and_save_snn.py --seed $seed --time $time --n_episodes $n_episodes --percentile $percentile \
    --epsilon $epsilon --game $game --normalize_on_spikes
else
     python convert_and_save_snn.py --seed $seed --time $time --n_episodes $n_episodes --percentile $percentile \
    --epsilon $epsilon --game $game
fi
exit
