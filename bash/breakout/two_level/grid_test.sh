#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 225 400 625
    do 
        for n_train in 16000
        do
            for n_test in 10000
            do
                for start_inhib in 1.0
                do
                    for max_inhib in 100.0
                    do
                        for p_low in 0.0 0.1 0.25 0.5
                        do
                            for norm in 40 50 60 70 80 90 100
                            do
                                for time in 250
                                do
                                    for theta_plus in 1.0 2.5
                                    do
                                        for theta_decay in 5e-5 1e-5
                                        do
                                            sbatch test.sh $seed $n_neurons $n_train $n_test $start_inhib \
                                                             $max_inhib $p_low $norm $time $theta_plus $theta_decay
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
