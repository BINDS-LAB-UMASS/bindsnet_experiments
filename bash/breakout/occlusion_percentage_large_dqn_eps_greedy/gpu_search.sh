#!/usr/bin/env bash

for seed in {0..99..1}
do
    for time in 200 250
    do
        for n_episodes in 1
        do
            for n_snn_episodes in 1
            do
                for percentile in 99.99
                do
                    for occlusion in {0..100..5}
                    do
                        sbatch gpu.sh $seed $time $n_episodes $n_snn_episodes $percentile $occlusion
                    done
                done
            done
        done
    done
done
