#!/usr/bin/env bash

for seed in 0
do
    for kernel_size in 18 20
    do
        for stride in 1 2 4
        do
            for n_filters in 25 36 49 64 81 100 121 144 169 196 225
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
                                                for norm in 0.25
                                                do
                                                    sbatch test.sh $seed $kernel_size $stride $n_filters $crop \
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
