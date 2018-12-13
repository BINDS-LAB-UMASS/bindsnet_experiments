#!/usr/bin/env bash

for seed in {0..99..1}
do
    for n_episodes in 1
    do
        for occlusion in {0..77..1}
        do
            sbatch submit.sh $seed $n_episodes $occlusion
        done
    done
done
