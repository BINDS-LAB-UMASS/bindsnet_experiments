for seed in 1
do
    for n_neurons in 100 200 300 400 500
    do 
        for n_train in 60000
        do
            for n_test in 10000
            do
                for inhib in 500
                do
                    for time in 300
                    do
                        for theta_plus in 0.05
                        do
                            for theta_decay in 1e-7
                            do
                                for p in $(seq 0.0 0.01 0.5)
                                do
                                    for intensity in 63.75
                                    do
                                        sbatch submit.sh $seed $n_neurons $n_train $n_test \
                                               $inhib $time $theta_plus $theta_decay $p \
                                               $intensity
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
