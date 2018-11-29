#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 100 200 300 400 500
    do 
        for n_train in 60000
        do
            for n_test in 10000
            do
                for inhib in 500
                do
                    for time in 250
                    do
                        for theta_plus in 0.05
                        do
                            for theta_decay in 1e-7
                            do
                                for intensity in 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75
                                do
                                    for lr in 1e-3
                                    do
                                        for lr_decay in 0.99
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
