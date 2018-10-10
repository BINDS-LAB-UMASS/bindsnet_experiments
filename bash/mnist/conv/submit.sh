#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/conv_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_train=${2:-60000}
n_test=${3:-10000}
kernel_size=${4:-8}
stride=${5:-4}
n_filters=${6:-25}
padding=${7:-0}
inhib=${8:-100.0}
time=${9:-350}
dt=${10:-1.0}
intensity=${11:-1.0}

cd ../../../experiments/mnist/
source activate py36

echo $seed $n_train $n_test $kernel_size $stride $n_filters $padding $inhib $time $dt $intensity

python conv.py --train --seed $seed --n_train $n_train --n_test $n_test --kernel_size $kernel_size \
			   --stride $stride --n_filters $n_filters --padding $padding --inhib $inhib --time $time \
			   --dt $dt --intensity $intensity
python conv.py --test --seed $seed --n_train $n_train --n_test $n_test --kernel_size $kernel_size \
               --stride $stride --n_filters $n_filters --padding $padding --inhib $inhib --time $time \
               --dt $dt --intensity $intensity
exit
