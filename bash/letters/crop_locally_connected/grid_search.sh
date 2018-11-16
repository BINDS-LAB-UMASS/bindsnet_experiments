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
    for kernel_size in 12 14 16
    do
        for stride in 2 4
        do
            for n_filters in 25 50 75 100
            do
                for crop in 0
                do
                    for lr in 1e-2
                    do
                        for lr_decay in 1 0.99
                        do
                            sbatch submit.sh $seed $kernel_size $stride $n_filters $crop $n_train $n_test $inhib \
                                             $time $theta_plus $theta_decay $intensity $norm $lr $lr_decay
                        done
                    done
                done
            done
        done
    done
done
