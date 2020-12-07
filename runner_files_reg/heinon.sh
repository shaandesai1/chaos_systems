#!/bin/bash

#mass spring
python main.py -ni 10000 -n_test_traj 1 -n_train_traj 50 -tmax 4.005 -dt 0.005 -dname heinon -noise_std 0 -batch_size 200 -learning_rate 1e-3
