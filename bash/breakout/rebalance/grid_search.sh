#!/usr/bin/env bash

for seed in 11 12 13 14 15
do
    for n_neurons in 100 200 300 400 500
    do 
        for n_train in 16000
        do
            for n_test in 10000
            do
                for inhib in 100
                do
                    for time in 250
                    do
                        for theta_plus in 10.0 25.0
                        do
                            for theta_decay in 5e-6 1e-5 5e-5
                            do
                                for norm in 37.5 50.0 62.5 75.0
                                do
                                    sbatch submit.sh $seed $n_neurons $n_train $n_test $inhib \
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
