#!/bin/bash

#mass spring
python main.py -ni 1000 -n_test_traj 25 -n_train_traj 25 -tmax 1.01 -dt 0.01 -dname painleve_I -noise_std 0 -type 1
