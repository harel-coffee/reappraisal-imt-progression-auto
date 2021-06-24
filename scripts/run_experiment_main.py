#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run the predictive analyses

"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
from os.path import join as opj
from pathlib import Path
import shutil
from tempfile import mkdtemp
from tqdm import tqdm

from sklearn.model_selection import LeaveOneOut
from my_sklearn_tools.model_selection import StratifiedKFoldReg
from my_sklearn_tools.pca_regressors import LassoPCR

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

def find_alpha_range(X, y, n_alphas=1000):

    from sklearn.linear_model._coordinate_descent import _alpha_grid
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA

    X_transform = PCA().fit_transform(VarianceThreshold().fit_transform(X))
    alphas = _alpha_grid(X = X_transform, y = y,n_alphas = n_alphas)

    return alphas

def run(X, y, cv_outer = LeaveOneOut(), n_alphas = 1000):

    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Lasso


    # Find alpha range
    alphas = find_alpha_range(X, y, n_alphas = n_alphas)

    list_y_pred = []
    list_y_true = []
    list_models = []

    for train_index, test_index in tqdm(cv_outer.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        list_y_true.append(y_test)

        cv_inner = StratifiedKFoldReg(n_splits=5, shuffle=True, random_state=0)

        lasso_pcr = LassoPCR(scale=False, cv=cv_inner, n_jobs=-1, alphas=alphas, lasso_kws = {'max_iter':1e6}, scoring="neg_mean_squared_error")
        lasso_pcr.fit(X_train, y_train)
        list_models.append(lasso_pcr)

        y_pred = lasso_pcr.predict(X_test)
        list_y_pred.append(y_pred)

    y_pred = np.concatenate(list_y_pred)
    y_true = np.concatenate(list_y_true)

    return y_pred, y_true, list_models

def run_transform(X, y,  transform, cv_outer = LeaveOneOut(), n_alphas = 1000):

    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Lasso
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import PowerTransformer

    y_trans = PowerTransformer(method=transform).fit_transform(y[:, None]).flatten()
    alphas = find_alpha_range(X, y_trans, n_alphas = n_alphas)

    list_y_pred = []
    list_y_true = []
    list_models = []

    for train_index, test_index in tqdm(cv_outer.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        list_y_true.append(y_test)

        cv_inner = StratifiedKFoldReg(n_splits=5, shuffle=True, random_state=0)

        lasso_pcr = LassoPCR(scale=False, cv=cv_inner, n_jobs=-1, alphas=alphas, lasso_kws = {'max_iter':1e6}, scoring="neg_mean_squared_error")

        regr_trans = TransformedTargetRegressor(regressor=lasso_pcr,
                                                transformer=PowerTransformer(method=transform))

        regr_trans.fit(X_train, y_train)
        list_models.append(regr_trans)

        y_pred = regr_trans.predict(X_test)
        list_y_pred.append(y_pred)

    y_pred = np.concatenate(list_y_pred)
    y_true = np.concatenate(list_y_true)

    return y_pred, y_true, list_models

def save_data(output_dir, y_pred, y_true, list_models):

    from joblib import dump

    y_preds_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
    y_preds_df.to_csv(opj(output_dir, "y_preds.csv"), index=False)

    # Save models
    Path(opj(output_dir, "models")).mkdir(exist_ok=True)

    for fold_id, model in enumerate(list_models):
        dump(model, opj(output_dir, 
                        "models", 
                        "fold_%.3d.joblib" % (fold_id+1)))

def main():

    parser = argparse.ArgumentParser(description='Run a particular experiment')
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
    opts = parser.parse_args()

    if opts.transform:
        msg = "Experiment to predict %s transformed %s from %s contrast maps with %d alphas" % (opts.transform, 
                                                                                                opts.target_var, 
                                                                                                opts.con_type, 
                                                                                                opts.n_alphas)
        print(msg)
    else:
        msg = "Experiment to predict untransformed %s from %s contrast maps with %d alphas" % (opts.target_var, 
                                                                                               opts.con_type, 
                                                                                               opts.n_alphas)
        print(msg)

    data_dir = os.path.abspath(opts.data_dir)

    if Path(data_dir).exists() is False:
        raise print("input directory does not exist")

    # Load data
    print("Loading data...")
    X, y = load_data(data_dir, opts.con_type, opts.target_var)

    # Build classifier
    cv_outer = LeaveOneOut()

    print("Running experiment...")
    if opts.transform:
        y_pred, y_true, list_models = run_transform(X, y,
                                                    opts.transform,
                                                    cv_outer=cv_outer,
                                                    n_alphas = opts.n_alphas)
    else:
        y_pred, y_true, list_models = run(X, y,
                                          cv_outer=cv_outer,
                                          n_alphas = opts.n_alphas)

    from sklearn.metrics import r2_score, mean_squared_error

    r = np.corrcoef(y_true, y_pred)[0,1]
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print("experiment gives r =%.3f, R2 = %.3f, MSE = %.3f" % (r, r2, mse))

    print("Saving results...")
    # Create output_directory for the given case (target->Input)
    if opts.transform:
        output_dir = opj(opts.output_dir, opts.transform + "_" + opts.target_var, opts.con_type)
    else:
        output_dir = opj(opts.output_dir, opts.target_var, opts.con_type)

    output_dir = os.path.abspath(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_data(output_dir, y_pred, y_true, list_models)

if __name__ == "__main__":
    sys.exit(main())
