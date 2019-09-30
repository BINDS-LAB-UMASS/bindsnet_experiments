#!/usr/bin/env bash

n_epochs=1
batch_size=32
inhib=250
time=100
theta_plus=0.05
tc_theta_decay=1e-7
norm=0.2

for seed in 0
do
    for kernel_size in 16
    do
        for stride in 2
        do
            for n_filters in 100
            do
                for crop in 0
                do
                    for lr_post in 1e-2
                    do
                        for lr_decay in 1
                        do
                            for intensity in 5
                            do
                                sbatch gpu.sh $seed $n_epochs $batch_size $inhib $kernel_size $stride $n_filters $crop\
                                 $lr_post $lr_decay $time $theta_plus $tc_theta_decay $intensity $norm
                            done
                        done
                    done
                done
            done
        done
    done
done
