#!/bin/bash

#mass spring
python main.py -ni 11000 -n_test_traj 1 -n_train_traj 25 -tmax 6.05 -dt 0.05 -dname mass_spring -noise_std 0 -batch_size 400 -learning_rate 1e-3
