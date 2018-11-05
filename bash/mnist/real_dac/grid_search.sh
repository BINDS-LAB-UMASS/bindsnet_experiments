#!/usr/bin/env bash

inhib=250
theta_plus=0.05
theta_decay=1e-7

for seed in 0
do
    for n_neurons in 100 250 500 750 1000
    do 
        for n_train in 60000
        do
            for n_test in 10000
            do
                for time in 25 50 100 250
                do
                    for lr in 0.1 0.05 0.01 0.005
                    do
                        for lr_decay in 1.0 0.99 0.95
                        do
                            sbatch submit.sh $seed $n_neurons $n_train $n_test $inhib $time \
                                             $lr $lr_decay $theta_plus $theta_decay
                        done
                    done
                done
            done
        done
    done
done
