#!/usr/bin/env bash

for seed in 0 1 2 3 4
do
    for p_remove in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
        sbatch gpu.sh $seed $p_remove
    done
done
