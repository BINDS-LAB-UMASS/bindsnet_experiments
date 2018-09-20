#!/usr/bin/env bash

for seed in 0
do
    for kernel_size in 8 12 16
    do
        for stride in 2 4
        do
            for n_filters in 25 49 100 144
            do
                for crop in 2 3 4 5
                do
                    for n_train in 60000
                    do
                        for n_test in 10000
                        do
                            for inhib in 250
                            do
                                for time in 300
                                do
                                    for theta_plus in 0.05
                                    do
                                        for theta_decay in 1e-7
                                        do
                                            for intensity in 0.5
                                            do
                                                for norm in 0.1 0.2 0.3
                                                do
                                                    sbatch submit.sh $seed $kernel_size $stride $n_filters $crop \
                                                                     $n_train $n_test $inhib $time $theta_plus \
                                                                     $theta_decay $intensity $norm
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
    done
done
