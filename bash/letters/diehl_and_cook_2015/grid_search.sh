for seed in 0
do
    for n_neurons in 100 225 400 625 900 1225
    do 
        for n_train in 60000
        do
            for n_test in 10000
            do
                for inhib in 500
                do
                    for time in 250
                    do
                        for theta_plus in 0.05
                        do
                            for theta_decay in 1e-7
                            do
                                for intensity in 0.5
                                do
                                        sbatch submit.sh $seed $n_neurons $n_train $n_test $inhib \
                                                         $time $theta_plus $theta_decay $intensity
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
