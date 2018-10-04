#!/usr/bin/env bash

for seed in 0 1 2 3 4
do
    for n_train in 60000
    do
        for n_test in 10000
        do
            for time in 10 25 50 75 100
            do
                for lr in 0.001 0.0025 0.005 0.0075 0.01
                do
                    for lr_decay in 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98
                    do
                        for update_interval in 500
                        do
                            for max_prob in 1.0
                            do
                                sbatch submit.sh $seed $n_train $n_test $time $lr $lr_decay $update_interval $max_prob
                            done
                        done
                    done
                done
            done
        done
    done
done
