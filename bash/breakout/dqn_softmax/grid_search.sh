#!/usr/bin/env bash

for seed in 0
do
    for time in 250
    do
        for n_episodes in 100
        do
            for n_snn_episodes in 100
            do
                for percentile in 98 98.5 99 99.5 99.9 100
                do
                    sbatch submit.sh $seed $time $n_episodes $n_snn_episodes $percentile
                done
            done
        done
    done
done
