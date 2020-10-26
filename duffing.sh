#!/bin/bash

#mass spring
python main.py -ni 300 -n_test_traj 50 -n_train_traj 25 -tmax 10.01 -dt 0.01 -dname duffing -noise_std 0 -type 4
