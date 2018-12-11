#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../../output/dac_synapses_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
p_destroy=${2:-0.1}

cd ../../../../experiments/robustness/mnist/
echo $seed $p_destroy
python dac_synapses.py --seed $seed --p_destroy $p_destroy
exit
