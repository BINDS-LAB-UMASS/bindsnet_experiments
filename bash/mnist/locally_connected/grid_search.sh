for seed in 6
do
    for kernel_size in 8 10 12 14 16
    do
        for stride in 2 4
        do
            for n_filters in 9 16 25 36 49
            do
                for n_train in 60000
                do
                    for n_test in 10000
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
                                            sbatch submit.sh $seed $kernel_size $stride \
                                                   $n_filters $n_train $n_test $inhib $time \
                                                   $theta_plus $theta_decay $intensity
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
