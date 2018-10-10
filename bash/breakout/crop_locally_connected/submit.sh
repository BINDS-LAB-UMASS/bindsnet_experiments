#!/usr/bin/env bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/crop_locally_connected_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
k1=${2:-25}
k2=${3:-36}
s1=${4:-5}
s2=${5:-6}
n_filters=${6:-25}
n_train=${7:-16000}
n_test=${8:-10000}
inhib=${9:-25.0}
time=${10:-350}
theta_plus=${11:-0.05}
theta_decay=${12:-1e-7}
norm=${13:-0.2}

cd ../../../experiments/breakout/
source activate py36

echo $seed $k1 $k2 $s1 $s2 $n_filters $n_train $n_test $inhib $time $theta_plus $theta_decay $norm

python crop_locally_connected.py --train --seed $seed --kernel_size $k1 $k2 --stride $s1 $s2 \
                                 --n_filters $n_filters --n_train $n_train --n_test $n_test --inhib $inhib \
                                 --time $time --theta_plus $theta_plus --theta_decay $theta_decay --norm $norm
python crop_locally_connected.py --test --seed $seed --kernel_size $k1 $k2 --stride $s1 $s2 \
                                 --n_filters $n_filters --n_train $n_train --n_test $n_test --inhib $inhib \
                                 --time $time --theta_plus $theta_plus --theta_decay $theta_decay --norm $norm
exit
