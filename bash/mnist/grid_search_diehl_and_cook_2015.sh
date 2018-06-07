for seed in 0
do
	for n_neurons in 100 400 
	do 
		for n_train in 30000
		do
			for n_test in 10000
			do
				for excite in 20 25 30 35 40 45 50 60 70 80 90 100
				do
					for inhib in 20 25 30 35 40 45 50 60 70 80 90 100
					do
						for time in 350
						do
							for theta_plus in 0.05 0.1 0.25 0.5 1.0
							do
								for theta_decay in 1e-7 1e-6 1e-5 1e-4
								do
									for intensity in 0.25 0.5 0.75 1.0
									do
										sbatch submit_diehl_and_cook_2015.sh $seed $n_neurons $n_train $n_test \
											   $excite $inhib $time $theta_plus $theta_decay $intensity
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
