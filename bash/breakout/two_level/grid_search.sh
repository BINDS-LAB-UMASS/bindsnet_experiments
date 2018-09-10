#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 100 225 400 625
    do 
        for n_train in 16000
        do
            for n_test in 10000
            do
                for start_inhib in 0.1 1.0 2.5
                do
                    for max_inhib in 100.0
                    do
                        for p_low in 0.1 0.25
                        do
                            for time in 250
                            do
                                for theta_plus in 0.05 0.5 1.0
                                do
                                    for theta_decay in 5e-6 1e-6 5e-5
                                    do
                                        sbatch submit.sh $seed $n_neurons $n_train $n_test $start_inhib \
                                                         $max_inhib $p_low $time $theta_plus $theta_decay
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
