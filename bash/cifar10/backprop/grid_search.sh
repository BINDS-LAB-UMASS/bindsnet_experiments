#!/usr/bin/env bash

for seed in 0
do
    for n_train in 60000
    do
        for n_test in 10000
        do
            for time in 10 50 100
            do
                for lr in 1e-3 5e-2 1e-2
                do
                    for lr_decay in 0.9 0.95 0.99
                    do
                        for wmin in -10 -1 -0.1
                        do
                            for wmax in 0.1 1 10
                            do
                                for norm in 50 100 250
                                do
                                    for update_interval in 500
                                    do
                                        sbatch submit.sh $seed $n_train $n_test $time $lr $lr_decay \
                                                         $wmin $wmax $norm $update_interval
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
