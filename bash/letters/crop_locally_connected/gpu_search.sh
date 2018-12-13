#!/usr/bin/env bash

n_train=120000
n_test=10000
inhib=250
time=250
theta_plus=0.05
theta_decay=1e-7
norm=0.2

for seed in 0 1 2 3 4
do
    for kernel_size in 12 16
    do
        for stride in 2 4
        do
            for n_filters in 25 50 100 250 500 1000
            do
                for crop in 4
                do
                    for lr in 1e-2
                    do
                        for lr_decay in 0.99
                        do
                            for intensity in 0.5
                            do
                                sbatch gpu.sh $seed $kernel_size $stride $n_filters $crop $n_train $n_test $inhib \
                                              $time $theta_plus $theta_decay $intensity $norm $lr $lr_decay
                            done
                        done
                    done
                done
            done
        done
    done
done
