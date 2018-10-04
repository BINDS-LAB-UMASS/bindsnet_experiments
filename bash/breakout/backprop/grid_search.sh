#!/usr/bin/env bash

for seed in 0 1 2
do
    for n_train in 60000
    do
        for n_test in 10000
        do
            for time in 10 50 100
            do
                for lr in 0.001 0.005 0.01
                do
                    for lr_decay in 0.9 0.95 0.99 1.0
                    do
                        for update_interval in 500
                        do
                            for max_prob in 1.0
                            do
                                for wmin in -10 -5 -1 -0.1
                                do
                                    for wmax in 0.1 1 5 10
                                    do
                                        for norm in 100 250 500 1000
                                        do
                                            sbatch submit.sh $seed $n_train $n_test $time $lr $lr_decay \
                                                             $update_interval $max_prob $wmin $wmax $norm
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
