#!/bin/bash

#mass spring
python main.py -ni 1000 -n_test_traj 25 -n_train_traj 25 -tmax 6.1 -dt 0.1 -dname mass_spring -noise_std 0.1
