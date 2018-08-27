#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 100 200 300
    do 
        for n_train in 6850
        do
            for n_test in 6850
            do
                for inhib in 100
                do
                    for time in 50 100 150 250
                    do
                        for theta_plus in 0.05 0.25 0.625 1.0
                        do
                            for theta_decay in 1e-7 5e-6 1e-6
                            do
                                sbatch submit.sh $seed $n_neurons $n_train $n_test $inhib $time $theta_plus $theta_decay
                            done
                        done
                    done
                done
            done
        done
    done
done
