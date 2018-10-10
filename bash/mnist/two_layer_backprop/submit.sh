#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/two_layer_backprop_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_hidden=${2:-100}
n_train=${3:-60000}
n_test=${4:-10000}
time=${5:-50}
lr=${6:-0.01}
lr_decay=${7:-0.99}
update_interval=${8:-500}

cd ../../../experiments/mnist/
source activate py36

echo $seed $n_hidden $n_train $n_test $time $lr $lr_decay $update_interval

python two_layer_backprop.py --train --seed $seed --n_hidden $n_hidden --n_train $n_train --n_test $n_test --time $time \
                             --lr $lr --lr_decay $lr_decay --update_interval $update_interval
python two_layer_backprop.py --test --seed $seed --n_hidden $n_hidden --n_train $n_train --n_test $n_test --time $time \
                             --lr $lr --lr_decay $lr_decay --update_interval $update_interval
exit
