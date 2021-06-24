#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import argparse
import os
import sys
from time import time
from joblib import Parallel, delayed
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm

def load_input(data_dir, cont_type):
    try:
        input_data = np.load(opj(data_dir, "input_data.npz"))
    except:
        raise ValueError("Input data does not exist"
                         " in the folder provided")

    return input_data[cont_type]

def load_target(data_dir, target_var):
    try:
        df = pd.read_csv(opj(data_dir, "target_data.csv"))
    except:
        raise ValueError("target data does not exist"
                         " in the folder provided")

    assert target_var in df.columns

    return df.loc[:, target_var].to_numpy()


def load_data(data_dir, cont_type, target_var):

    X = load_input(data_dir, cont_type)
    y = load_target(data_dir, target_var)

    return X, y


def extract_map(pip, X):
    """
    Function to extract the phenotype maps

    """
    from sklearn.preprocessing import StandardScaler

    ss = StandardScaler(with_std=False) # We need this because we are using the mean substracted features
                                        # This is important for computing the encoding maps

    V = pip.named_steps['pca'].components_ # PCA transformation
    beta =  pip.named_steps['lasso'].coef_ # Coefficients in the PC space
    insert_voxels = pip.named_steps['variancethreshold'].inverse_transform
    filter_voxels = pip.named_steps['variancethreshold'].transform

    phenotypes_dict = dict()

    w = V.T @ beta
    w_dec = np.squeeze(insert_voxels(w[None, :]))
    phenotypes_dict['w_dec'] = w_dec

    X_mean = ss.fit_transform(X)
    X_vt = filter_voxels(X_mean) # Concentrate on the voxels used
    w_enc = X_vt.T @ (X_vt @ w)
    w_enc = np.squeeze(insert_voxels(w_enc[None,:]))
    phenotypes_dict['w_enc'] = w_enc

    return phenotypes_dict

def compute_phenotypes(X, y, random_state, n_alphas=1000, n_splits=5):

    from my_sklearn_tools.model_selection import StratifiedKFoldReg
    from my_sklearn_tools.pca_regressors import LassoPCR

    cv = StratifiedKFoldReg(n_splits=n_splits, random_state=random_state, shuffle=True)

    lasso_pcr = LassoPCR(scale=False, cv=cv, n_jobs=1, n_alphas=n_alphas, lasso_kws = {'max_iter':1e6}, scoring="neg_mean_squared_error")
    lasso_pcr.fit(X, y)

    pip_opt = lasso_pcr.best_estimator_
    phenotypes = extract_map(pip_opt, X)
    return phenotypes

def run_bootstrap(X, y, boot_id):
    from sklearn.utils import resample

    X_boot, y_boot = resample(X, y, random_state=boot_id) # set to boot_id for reproducbility
    phenotypes = compute_phenotypes(X_boot, y_boot, random_state=boot_id) # set to boot_id for reproducbility
    return phenotypes


def main():

    parser = argparse.ArgumentParser(description='Compute phenotypes, with bootstrapping')
    parser.add_argument('--data_dir', 
                        dest="data_dir", 
                        type=str, 
                        required=True, 
                        help='Directory where data are located')
    parser.add_argument('--output_dir',
                        dest="output_dir", 
                        type=str,
                        required=True,
                        help='Directory where we are going to save the resuts')
    parser.add_argument('--con_type',
                        dest="con_type",
                        type=str,
                        required=True,
                        choices=['look_neg_look_neut', 
                                 'reg_neg_look_neg'],
                        help='Which contrast maps to take as input')
    parser.add_argument('--target_var',
                        dest="target_var",
                        type=str,
                        required=True,
                        help='Which variable to take as target')
    parser.add_argument('--n_alphas',
                        dest="n_alphas",
                        type=int,
                        default=1000,
                        help='Number of alphas to try for optimization')
    parser.add_argument('--transform',
                        dest="transform",
                        type=str,
                        choices= ['yeo-johnson', 'box-cox'],
                        help='Transform target variable')
    parser.add_argument('--n_boots',
                        dest="n_boots",
                        type=int,
                        default=5000,
                        help="number of bootstraps")
    parser.add_argument('--n_jobs',
                        dest="n_jobs",
                        type=int,
                        default=1,
                        help="number of jobs for parallel bootstrapping")
    opts = parser.parse_args()

    if opts.transform:
        msg = "Computing phenotypes for case %s transformed %s from %s contrast maps with %d alphas and %d bootstraps" % (opts.transform,
                                                                                                opts.target_var,
                                                                                                opts.con_type,
                                                                                                opts.n_alphas,
                                                                                                opts.n_boots)
        print(msg)
    else:
        msg = "Computing phenotypes for case (untransformed) %s from %s contrast maps with %d alphas and %d bootstraps" % (opts.target_var,
                                                                                                opts.con_type,
                                                                                                opts.n_alphas,
                                                                                                opts.n_boots)
        print(msg)


    data_dir = os.path.abspath(opts.data_dir)

    if Path(data_dir).exists() is False:
        raise print("input directory does not exist")

    # Load data
    print("Loading data...")
    X, y = load_data(data_dir, opts.con_type, opts.target_var)

    n_alphas = opts.n_alphas
    n_boots = opts.n_boots
    n_jobs = opts.n_jobs
    RANDOM_STATE=0

    # Create output_directory for the given case (target->Input)
    if opts.transform:
        output_dir = opj(opts.output_dir, opts.transform + "_" + opts.target_var, opts.con_type)
    else:
        output_dir = opj(opts.output_dir, opts.target_var, opts.con_type)

    output_dir = os.path.abspath(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compute full data phenotypes
    res_full = compute_phenotypes(X, y, random_state=RANDOM_STATE, n_alphas=n_alphas)

    w_dec_full = res_full['w_dec']
    w_enc_full = res_full['w_enc']

    # Save these results
    np.save(opj(output_dir, "decoding_weights.npy"), w_dec_full)
    np.save(opj(output_dir, "encoding_weights.npy"), w_enc_full)

    #############
    ### BOOTSTRAPPING
    t_0 = time()
    res_boots = Parallel(n_jobs=n_jobs)(delayed(run_bootstrap)(X, y, boot_id) for boot_id in tqdm(np.arange(n_boots)))
    t_f = time()
    print("ellapsed time: %f s" % (t_f-t_0))
    print(res_boots)

    w_dec_boot = np.array([res['w_dec'] for res in res_boots])
    w_enc_boot = np.array([res['w_enc'] for res in res_boots])

    np.save(opj(output_dir, "decoding_weights_boots.npy"), w_dec_boot)
    np.save(opj(output_dir, "encoding_weights_boots.npy"), w_enc_boot)

if __name__ == "__main__":
    sys.exit(main())
