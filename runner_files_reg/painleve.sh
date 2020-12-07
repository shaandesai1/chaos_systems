#!/bin/bash

#mass spring
python main.py -ni 20000 -n_test_traj 1 -n_train_traj 25 -tmax 1.001 -dt 0.001 -dname painleve_I -noise_std 0 -type 1 -batch_size 200 -learning_rate 1e-2
