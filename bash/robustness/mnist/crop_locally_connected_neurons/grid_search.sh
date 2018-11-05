#!/usr/bin/env bash

n_train=60000
n_test=10000
inhib=250
time=250
theta_plus=0.05
theta_decay=1e-7
intensity=0.5
norm=0.2

for seed in 0
do
    for kernel_size in 10 12 14 16 18
    do
        for stride in 2 4
        do
            for n_filters in 25 50 75 100 125 150
            do
                for crop in 4
                do
                    for lr in 1e-1 5e-2 1e-2 5e-3
                    do
                        for lr_decay in 1 0.99
                        do
                            for p_remove in 0 0.1 0.25 0.5 0.75 0.9
                            do
                                sbatch submit.sh $seed $kernel_size $stride $n_filters $crop $n_train $n_test $inhib \
                                                 $time $theta_plus $theta_decay $intensity $norm $lr $lr_decay \
                                                 $p_remove
                            done
                        done
                    done
                done
            done
        done
    done
done
