# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:15:31 2019

@author: Adam
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
import pickle
import csv
from timeit import default_timer as timer
import os

MAX_EVALS = 5
N_FOLDS = 4
ITERATION = 0


def objective(params, n_folds=N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params["boosting_type"].get("subsample", 1.0)

    # Extract the boosting type
    params["boosting_type"] = params["boosting_type"]["boosting_type"]
    params["subsample"] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in [
        "num_leaves",
        "subsample_for_bin",
        "min_child_samples",
        "max_bin",
        "bagging_freq",
    ]:
        params[parameter_name] = int(params[parameter_name])

    start = timer()

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(
        params,
        train_set,
        nfold=n_folds,
        num_boost_round=10000,
        early_stopping_rounds=100,
        metrics="auc",
        seed=50,
        verbose_eval=100,
    )

    run_time = timer() - start

    # Extract the best score and std
    best_score = max(cv_results["auc-mean"])
    best_std = cv_results["auc-stdv"][np.argmax(cv_results["auc-mean"])]

    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results["auc-mean"]) + 1)

    # Write to the csv file ('a' means append)
    filename = "results/lgb_tuning_history.csv"
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([loss, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {
        "loss": loss,
        "params": params,
        "iteration": ITERATION,
        "estimators": n_estimators,
        "train_time": run_time,
        "status": STATUS_OK,
    }


def run_trials():

    trials_step = (
        1
    )  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5  # initial max_trials. put something small to not have to wait

    # Define the search space
    params = {
        "max_depth": 14,
        "save_binary": True,
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "drop_seed": 1337,
        "data_random_seed": 1337,
        "objective": "binary",
        "is_unbalance": True,
        "boost_from_average": False,
        "max_bin": 120,
        "min_sum_hessian_in_leaf": hp.loguniform(
            "min_sum_hessian_in_leaf", np.log(0.0001), np.log(0.2)
        ),
        "bagging_fraction": 1.0,
        "bagging_freq": hp.uniform("bagging_freq", 0, 10),
        "min_gain_to_split": hp.uniform(
            "min_gain_to_split", 0.0, 1.0
        ),  # used to control regularization
        "boosting_type": hp.choice(
            "boosting_type",
            [
                {
                    "boosting_type": "gbdt",
                    "subsample": hp.uniform("gdbt_subsample", 0.5, 1),
                }
            ],
        ),
        "num_leaves": hp.quniform("num_leaves", 5, 200, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        "subsample_for_bin": hp.quniform("subsample_for_bin", 20000, 300000, 20000),
        "min_child_samples": hp.quniform(
            "min_child_samples", 5, 500, 5
        ),  # min_data_in_leaf
        "lambda_l1": hp.uniform("lambda_l1", 0.0, 10.0),
        "lambda_l2": hp.uniform("lambda_l2", 0.0, 10.0),
        "feature_fraction": hp.uniform("feature_fraction", 0.6, 1.0),
    }

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("results/trials.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print(
            "Rerunning from {} trials to {} (+{}) trials".format(
                len(trials.trials), max_trials, trials_step
            )
        )
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=max_trials,
        trials=trials,
    )

    print("Best:", best)

    # save the trials object
    with open("results/trials.hyperopt", "wb") as f:
        pickle.dump(trials, f)


if __name__ == "__main__":
    # Read in data and separate into training and testing sets
    df_train = pd.read_csv("dataset/train.csv", sep=",", index_col="ID_code")

    train_features = df_train.drop(["target"], axis=1).values
    train_labels = df_train["target"].values

    train_set = lgb.Dataset(train_features, train_labels)

    # create file to save results
    filename = "results/lgb_tuning_history.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            # Write the headers to the file
            writer.writerow(["loss", "params", "iteration", "estimators", "train_time"])

    while True:
        run_trials()
