#!/bin/bash

python run_experiment_sensitivity.py --data_dir ../data --output_dir ../results/prediction_sensitivity --con_type look_neg_look_neut --target_var chg_LNeg_LNur > ratings_sensitivity_con_0001.out
python run_experiment_sensitivity.py --data_dir ../data --output_dir ../results/prediction_sensitivity --con_type reg_neg_look_neg --target_var chg_RNeg_LNeg > ratings_sensitivity_con_0002.out
