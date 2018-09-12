#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 200 300 400 500 600
    do 
        for n_train in 16000
        do
            for n_test in 10000
            do
                for inhib in 100
                do
                    for time in 250
                    do
                        for theta_plus in 0.1 0.5 1.0
                        do
                            for theta_decay in 5e-6 1e-6 5e-5
                            do
                                for norm in 40 50 60 70 80 90 100
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
