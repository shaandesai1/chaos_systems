#!/bin/bash

#mass spring
python main.py -ni 200 -n_test_traj 25 -n_train_traj 25 -tmax 10.01 -dt 0.01 -dname forced_mass_spring -noise_std 0 -type 2
