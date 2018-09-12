#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 100 200 300 400 500
    do
        for n_train in 10000 20000 30000 40000 50000
        do
            for n_test in 10000
            do
                for inhib in 100
                do
                    for time in 250
                    do
                        for theta_plus in 0.05
                        do
                            for theta_decay in 1e-7 5e-6 1e-6
                            do
                                for norm in 0.01 0.05 0.1 0.15 0.2
                                    sbatch gpu_submit.sh $seed $n_neurons $n_train $n_test $inhib \
                                                         $time $theta_plus $theta_decay $norm
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
