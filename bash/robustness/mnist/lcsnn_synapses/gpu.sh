#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#SBATCH --mem=12000
#SBATCH --account=rkozma
#SBATCH --output=../../../output/lcsnn_synapses_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
p_destroy=${2:-0.1}

cd ../../../../experiments/robustness/mnist/
echo $seed $p_destroy
python lcsnn_synapses.py --seed $seed --p_destroy $p_destroy
exit
