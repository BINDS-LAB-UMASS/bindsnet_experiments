#!/usr/bin/env bash

for seed in 0 1 2
do
    for n_hidden in 128 256 512 1024 
    do
        for n_train in 60000
        do
            for n_test in 10000
            do
                for time in 5 10 25 50 100 250 500
                do
                    for lr in 0.0125 0.015 0.016 0.017 0.018 0.019 0.02 0.0225
                    do
                        for lr_decay in 0.96 0.97 0.98
                        do
                            for update_interval in 500
                            do
                                sbatch submit.sh $seed $n_hidden $n_train $n_test $time $lr $lr_decay $update_interval
                            done
                        done
                    done
                done
            done
        done
    done
done
