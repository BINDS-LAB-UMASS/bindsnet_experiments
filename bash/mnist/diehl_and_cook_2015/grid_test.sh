for seed in 6
do
    for n_neurons in 400
    do 
        for n_train in 30000 60000
        do
            for n_test in 10000
            do
                for excite in 100
                do
                    for inhib in {300..750..50}
                    do
                        for time in 275 350 425 500
                        do
                            for theta_plus in 0.01 0.05 0.1
                            do
                                for theta_decay in 1e-7
                                do
                                    for intensity in 0.4 0.5 0.6
                                    do
                                        for X_Ae_decay in 0.25 0.5 0.75
                                        do
                                            sbatch test.sh $seed $n_neurons $n_train $n_test \
                                                   $excite $inhib $time $theta_plus $theta_decay \
                                                   $intensity $X_Ae_decay
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
