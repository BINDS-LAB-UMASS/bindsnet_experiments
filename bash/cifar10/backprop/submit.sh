#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/backprop_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_train=${2:-60000}
n_test=${3:-10000}
time=${4:-350}
lr=${5:-0.01}
lr_decay=${6:-0.99}
wmin=${7:--1.0}
wmax=${8:-1.0}
norm=${9:-250}
update_interval=${10:-500}

cd ../../../scripts/cifar10/
source activate py36

echo $seed $n_train $n_test $time $lr $lr_decay $wmin $wmax $norm $update_interval

python backprop.py --train --seed $seed --n_train $n_train --n_test $n_test --time $time --lr $lr \
                   --lr_decay $lr_decay --wmin $wmin --wmax $wmax --norm $norm \
                   --update_interval $update_interval
python backprop.py --test --seed $seed --n_train $n_train --n_test $n_test --time $time --lr $lr \
                   --lr_decay $lr_decay --wmin $wmin --wmax $wmax --norm $norm \
                   --update_interval $update_interval
exit
