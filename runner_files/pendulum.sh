#!/bin/bash

#mass spring
python main.py -ni 20000 -n_test_traj 1 -n_train_traj 25 -tmax 3.05 -dt 0.05 -dname pendulum -noise_std 0 -batch_size 200 -learning_rate 1e-3
