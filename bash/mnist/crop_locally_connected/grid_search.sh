#!/usr/bin/env bash

for seed in 0
do
    for kernel_size in 8 10 12 14 16
    do
        for stride in 2 4
        do
            for n_filters in 16 25 36 49 64 81 100
            do
                for crop in 2 4
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
                                                for norm in 0.1 0.15 0.2
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
