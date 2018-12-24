#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
time=${2:-250}
n_snn_episodes=${3:-1}
parameter1=${4:-1}
parameter2=${5:-1}
parameter3=${6:-1}
parameter4=${7:-1}
parameter5=${8:-1}


cd ../../../experiments/conversion/
echo $seed $time $n_snn_episodes $parameter1 $parameter2 $parameter3 $parameter4 $parameter5
python3 large_dqn_lif.py @@seed $seed @@time $time @@n_snn_episodes $n_snn_episodes\
                                         @@parameter1 $parameter1 @@parameter2 $parameter2\
                                         @@parameter3 $parameter3 @@parameter4 $parameter4\
                                         @@parameter5 $parameter5
exit
