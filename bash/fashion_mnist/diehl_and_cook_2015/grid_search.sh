#!/usr/bin/env bash

for seed in 0
do
    for n_neurons in 100 200 300 400 500
    do 
        for n_train in 60000
        do
            for n_test in 10000
            do
                for inhib in 100 250 500
                do
                    for time in 100 200 300 400 500
                    do
                        for theta_plus in 0.05
                        do
                            for theta_decay in 1e-7
                            do
                                for intensity in 0.5
                                do
                                    for norm in 50 75 100 125 150
                                    do
                                        sbatch submit.sh $seed $n_neurons $n_train $n_test \
                                               $inhib $time $theta_plus $theta_decay \
                                               $intensity $norm
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
