""""
Main for Gaussian Process Regression. Inpired by: https://juanitorduz.github.io/gaussian_process_time_series/
"""
from typing import Tuple
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import time
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor

from hms.optimization import optimize_with_hms
from utils import create_experiment_dir, load_data, get_kernel
from plotting import *


def predict(
    gpr: GaussianProcessRegressor,
    df: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: str
) -> Tuple:
    
    # Predict
    x = np.concatenate([x_train, x_test])
    y_mean, y_std = gpr.predict(x, return_std=True)

    # Calculate R2 scores
    r2_score_train = gpr.score(x_train, y_train)
    r2_score_test = gpr.score(x_test, y_test)

    # Add predictions to the dataframe
    df['time'] = df.index
    df['y_mean'] = y_mean
    df['y_std'] = y_std
    df['y_lwr'] = df['y_mean'] - 2*df['y_std']
    df['y_upr'] = df['y_mean'] + 2*df['y_std']

    # Plot the predictions
    plot_predictions(
        df=df,
        n_train=x_train.shape[0],
        r2_score_test=r2_score_test,
        results_dir=results_dir
    )    

    # Plot the errors
    plot_errors(
        gpr=gpr,
        x_test=x_test,
        y_test=y_test,
        results_dir=results_dir
    )
    
    return df, r2_score_train, r2_score_test

def gpr(
    data_path: str,
    column: str,
    split: float,
    kernel_path: str,
    optimizer: str,
    results_dir: str,
    verbose: bool = False
):

    # Load data
    df, x_train, x_test, y_train, y_test, mean, std = load_data(
        data_path=data_path,
        column=column,
        split=split,
        verbose=verbose
    )

    # Define the kernel
    kernel = get_kernel(kernel_path=kernel_path)

    # Define GaussianProcessRegressor object. 
    gpr = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=10, 
        normalize_y=True,
        alpha=0.0,
    )

    # Plot samples from prior
    if verbose:
        plot_samples(
            gpr=gpr,
            x_train=x_train,
            y_train=y_train,
            mean=mean,
            std=std,
            n_samples=10,
            normalize=True,
            title='Prior Samples'
        )

    # Fit the model
    print("[info] Fitting the model...")
    start_time = time.time()
    if optimizer == "sklearn":
        gpr.fit(x_train, y_train)
    elif optimizer == "hms":
        gpr = optimize_with_hms(gpr, x_train, y_train)
    print(f"[info] Fitting the model took {time.time() - start_time:.2f} seconds")
    
    # Plot samples from posterior
    if verbose:
        plot_samples(
            gpr=gpr,
            x_train=x_train,
            y_train=y_train,
            mean=mean,
            std=std,
            n_samples=10,
            normalize=False,
            title='Fitted Posterior Samples'
        )
    # Predict
    df, r2_score_train, r2_score_test = predict(
        gpr=gpr,
        df=df,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        results_dir=results_dir
    )
    print(f"[info] Train R2 score: {r2_score_train:.3f}")
    print(f"[info] Test R2 score:  {r2_score_test:.3f}")


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-k", "--kernel_path", type=str, required=True,
                        help="Path to the kernel configuration .py file")
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Path to the data file time series")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train-test split ratio")
    parser.add_argument("-o", "--optimizer", type=str, default="sklearn", choices=["sklearn", "hms"],
                        help="Optimization method to use for the kernel hyperparameters optimization")
    parser.add_argument("--column", type=str, default="Close",
                        help="Dataframe column to use for the time series prediction")
    parser.add_argument("-r", "--results_dir", type=str, default="results",
                        help="Directory to save the results")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode. Plot the data, print info, etc.")
    args = parser.parse_args()

    # Create experiment directory
    args.results_dir = create_experiment_dir(args.results_dir, args.kernel_path)

    # Run the Gaussian Process Regression
    gpr(**args.__dict__)
