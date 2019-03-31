#!/usr/bin/env bash

for seed in 0
do
    for time in 200 250
    do
        for n_episodes in 1 3
        do
            for n_snn_episodes in 100
            do
                for percentile in 99.99
                do
                    for epsilon in 0.05
                    do
                        for game in 'video_pinball'
                        do
                            sbatch gpu.sh $seed $time $n_episodes $n_snn_episodes $percentile $epsilon $game
                        end
                    done
                done
            done
        done
    done
done
