#!/usr/bin/env bash

for seed in 0 1 2
do
    for n_hidden in 32 64 128 256 512
    do
        for n_train in 60000
        do
            for n_test in 10000
            do
                for time in 10 25 50
                do
                    for lr in 0.0075 0.01 0.025
                    do
                        for lr_decay in 0.9 0.95 0.99 0.995 1.0
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
