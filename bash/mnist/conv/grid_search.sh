for seed in 1
do
    for n_train in 60000
    do
		for n_test in 10000
        do
			for kernel_size in 4 6 8 10 12 14 16
			do
				for stride in 2 4
				do
					for n_filters in 9 16 25 36 49 64 81 100
					do
						for padding in 0
						do
							for inhib in 100
							do
								for time in 25 50 100 250
								do
									for dt in 1.0
									do
										for intensity in 0.5
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
    done
done
