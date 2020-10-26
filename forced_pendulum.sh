#!/bin/bash

#mass spring
python main.py -ni 200 -n_test_traj 25 -n_train_traj 25 -tmax 20.1 -dt 0.1 -dname forced_pendulum -noise_std 0 -type 4
