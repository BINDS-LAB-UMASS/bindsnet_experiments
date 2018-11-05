#!/usr/bin/env bash

padding=0
inhib=250
dt=1
intensity=0.5

for seed in 0
do
    for n_train in 60000
    do
		for n_test in 10000
        do
			for kernel_size in 4 6 8 10 12 14
			do
				for stride in 2
				do
					for n_filters in 25 50 100 150
					do
                        for time in 25 50 100
                        do
                            for lr in 0.01 0.005 0.001
                            do
                                for lr_decay in 1.0 0.99 0.975 0.95
                                do
                                    sbatch submit.sh $seed $n_train $n_test $kernel_size $stride $n_filters \
                                                     $padding $inhib $time $dt $intensity
                                done
                            done
                        done
                	done
				done
            done
        done
    done
done
