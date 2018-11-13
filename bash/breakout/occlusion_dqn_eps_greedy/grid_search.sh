#!/usr/bin/env bash

for seed in 0 1 2 3 4
do
    for time in 100
    do
        for n_episodes in 100
        do
            for n_snn_episodes in 100
            do
                for percentile in 98 99 99.5
                do
                    for occlusion in {0..77..1}
                    do
                        sbatch submit.sh $seed $time $n_episodes $n_snn_episodes $percentile $occlusion
                    done
                done
            done
        done
    done
done
