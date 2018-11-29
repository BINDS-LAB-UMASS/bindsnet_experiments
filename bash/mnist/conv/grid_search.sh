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
			for kernel_size in 6 8 10
			do
				for stride in 2 4
				do
					for n_filters in 10 25 50
					do
                        for time in 10 25 50
                        do
                            for lr in 0.01 0.005 0.0025 0.001
                            do
                                for lr_decay in 0.995
                                do
                                    sbatch submit.sh $seed $n_train $n_test $kernel_size $stride $n_filters \
                                                     $padding $inhib $time $dt $intensity $lr $lr_decay
                                done
                            done
                        done
                	done
				done
            done
        done
    done
done
