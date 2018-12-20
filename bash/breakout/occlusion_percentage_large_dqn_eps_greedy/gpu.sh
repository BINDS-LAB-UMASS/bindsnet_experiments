#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=../../output/occlusion_large_dqn_eps_greedy_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
time=${2:-200}
n_episodes=${3:-1}
n_snn_episodes=${4:-100}
percentile=${5:-99.99}
occlusion=${6:-0}

cd ../../../experiments/conversion/
echo $seed $time $n_episodes $n_snn_episodes $percentile $occlusion
python occlusion_large_dqn_eps_greedy.py --seed $seed --time $time --n_episodes $n_episodes \
                                         --n_snn_episodes $n_snn_episodes --percentile $percentile \
                                         --occlusion $occlusion
exit
