#!/bin/bash

python run_compute_phenotypes.py --data_dir ../data --output_dir ../results/phenotypes --con_type look_neg_look_neut --target_var chg_LNeg_LNur --n_boots 5000 --n_jobs 10 > ratings_phenotypes_001.out
python run_compute_phenotypes.py --data_dir ../data --output_dir ../results/phenotypes --con_type reg_neg_look_neg --target_var chg_RNeg_LNeg --n_boots 5000 --n_jobs 10 > ratings_phenotypes_002.out

