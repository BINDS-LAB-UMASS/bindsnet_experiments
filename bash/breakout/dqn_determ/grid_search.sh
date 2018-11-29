#!/usr/bin/env bash

for seed in 0
do
    for time in 100 250 500
    do
        for n_episodes in 100
        do
            for percentile in 98 98.5 99 99.5 99.9 100
            do
                sbatch submit.sh $seed $time $n_episodes $percentile
            done
        done
    done
done
