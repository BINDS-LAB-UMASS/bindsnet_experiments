#!/usr/bin/env bash
filename="particle_pos.txt"

time=250

while IFS='' read -r weight1 weight2 weight3 weight4 weight5; do
    for time in 250
    do
        for seed in {0..4..1}
        do
            for n_snn_episodes in 1
            do
                sbatch gpu.sh $seed $time $n_snn_episodes $weight1 $weight2 $weight3 $weight4 $weight5
            done
        done
    done
done < "$filename"