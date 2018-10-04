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
update_interval=${7:-500}
max_prob=${8:-1.0}

cd ../../../scripts/fashion_mnist/
source activate py36

echo $seed $n_train $n_test $time $lr $lr_decay $update_interval $max_prob

python backprop.py --train --seed $seed --n_train $n_train --n_test $n_test --time $time --lr $lr \
                   --lr_decay $lr_decay --update_interval $update_interval --max_prob $max_prob
python backprop.py --test --seed $seed --n_train $n_train --n_test $n_test --time $time --lr $lr \
                   --lr_decay $lr_decay --update_interval $update_interval --max_prob $max_prob
exit
