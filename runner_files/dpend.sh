#!/bin/bash

#mass spring
python main.py -ni 100000 -n_test_traj 3 -n_train_traj 100 -tmax 10.01 -dt 0.01 -dname dpend -noise_std 0 -batch_size 200 -learning_rate 1e-3
