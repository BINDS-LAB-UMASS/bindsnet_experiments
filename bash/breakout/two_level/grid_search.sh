#!/usr/bin/env bash

for seed in 11 12 13 14 15
do
    for n_neurons in 100 200 300 400 500
    do 
        for n_train in 16000
        do
            for n_test in 10000
            do
                for start_inhib in 0.5 1.0 2.5
                do
                    for max_inhib in 100.0
                    do
                        for p_low in 0.0 0.25 0.5 1.0
                        do
                            for norm in 62.5
                            do
                                for time in 250
                                do
                                    for theta_plus in 5.0
                                    do
                                        for theta_decay in 1e-5
                                        do
                                            sbatch submit.sh $seed $n_neurons $n_train $n_test $start_inhib \
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
