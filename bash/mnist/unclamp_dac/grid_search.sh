#!/usr/bin/env bash

for seed in 0 1 2 3 4
do
    for n_neurons in 100 250 500
    do 
        for n_train in 60000
        do
            for n_test in 10000
            do
                for inhib in 250
                do
                    for time in 50 100
                    do
                        for theta_plus in 0.05
                        do
                            for theta_decay in 1e-7
                            do
                                for intensity in 4
                                do
                                    for lr in 0.01 0.005
                                    do
                                        for lr_decay in 1 0.99
                                        do
                                            sbatch submit.sh $seed $n_neurons $n_train $n_test $inhib \
                                                             $time $theta_plus $theta_decay $intensity \
                                                             $lr $lr_decay
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
