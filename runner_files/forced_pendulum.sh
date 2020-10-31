#!/bin/bash

#mass spring
python main.py -ni 10000 -n_test_traj 1 -n_train_traj 50 -tmax 10.01 -dt 0.01 -dname forced_pendulum -noise_std 0 -batch_size 500 -learning_rate 1e-3
