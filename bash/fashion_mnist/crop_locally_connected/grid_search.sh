#!/usr/bin/env bash

n_train=60000
n_test=10000
inhib=250
time=250

for seed in 0
do
    for kernel_size in 10 12 14 16
    do
        for stride in 2 4
        do
            for n_filters in 25
            do
                for crop in 0
                do
                    for lr in 1e-2 5e-3
                    do
                        for lr_decay in 1 0.99
                        do
							for theta_plus in 0.05 1
							do
								for theta_decay in 1e-7 1e-6
								do
									for norm in 0.1 0.2 0.3
									do
                            			sbatch submit.sh $seed $kernel_size $stride $n_filters $crop $n_train $n_test $inhib \
                                            			 $time $theta_plus $theta_decay $norm $lr $lr_decay
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
