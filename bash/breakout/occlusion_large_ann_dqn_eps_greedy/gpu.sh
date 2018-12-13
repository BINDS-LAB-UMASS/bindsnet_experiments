#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=05-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=../../output/occlusion_large_ann_dqn_eps_greedy_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_episodes=${2:-1}
occlusion=${3:-0}

cd ../../../experiments/conversion/
echo $seed $n_episodes $occlusion
python occlusion_large_ann_dqn_eps_greedy.py --seed $seed --n_episodes $n_episodes --occlusion $occlusion
exit
