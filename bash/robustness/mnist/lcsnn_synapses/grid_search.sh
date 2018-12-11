#!/usr/bin/env bash

for seed in 0 1 2 3 4
do
    for p_destroy in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
        sbatch submit.sh $seed $p_destroy
    done
done
