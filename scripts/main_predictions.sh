#!/bin/bash

python run_experiment_main.py --data_dir ../data --output_dir ../results/prediction --con_type look_neg_look_neut --target_var mavgccaf_chgPerYr > main_experiment_con_0001_avg_prog.out
python run_experiment_main.py --data_dir ../data --output_dir ../results/prediction --con_type reg_neg_look_neg --target_var mavgccaf_chgPerYr > main_experiment_con_0002_avg_prog.out
python run_experiment_main.py --data_dir ../data --output_dir ../results/prediction --con_type look_neg_look_neut --target_var Site_of_Fastest_IMT_Max_Progr --transform yeo-johnson > main_experiment_con_0001_max_progt.out
python run_experiment_main.py --data_dir ../data --output_dir ../results/prediction --con_type reg_neg_look_neg --target_var Site_of_Fastest_IMT_Max_Progr --transform yeo-johnson > main_experiment_con_0002_max_prog.out
