#!/usr/bin/env bash

for seed in 0
do
    for kernel_size in '25 36' '10 36' '10 24' '25 24'
    do
        for stride in '5 6' '10 12' '5 12' '10 6'
        do
            for n_filters in 9 16 25
            do
                for n_train in 16000
                do
                    for n_test in 10000
                    do
                        for inhib in 100
                        do
                            for time in 250
                            do
                                for theta_plus in 1.0
                                do
                                    for theta_decay in 1e-5
                                    do
                                        for norm in 0.01 0.05 0.1 0.15 0.2
                                        do
                                            sbatch gpu_test.sh $seed $kernel_size $stride $n_filters $n_train \
                                                               $n_test $inhib $time $theta_plus $theta_decay $norm
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
