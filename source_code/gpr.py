""""
Main for Gaussian Process Regression. Inpired by: https://juanitorduz.github.io/gaussian_process_time_series/
"""
from typing import Tuple, Callable
from argparse import ArgumentParser

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
from datetime import datetime
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error

# Set plotting style
sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100


def create_experiment_dir(dir_path: str, kernel_path: str) -> str:

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create directory
    dir_path = os.path.join(dir_path, timestamp)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Copy configuration file to experiment directory
    os.system(f"cp {kernel_path} {os.path.join(dir_path, 'kernel.py')}")
    
    # Return
    return dir_path


def load_data(data_path: str, column: str, split: float = 0.8, verbose: bool = False) -> Tuple:

    # Load the data
    df = pd.read_csv(data_path)
    mean = df[column].mean()
    std = df[column].std()

    # Create training and testing sets
    n = len(df)
    n_train = int(n * split)

    # Convert data to numpy arrays
    x = df.index.values.reshape(-1, 1)
    y = df[column].values.reshape(n, 1)

    # Split data into training and test sets
    x_train = x[:n_train]
    x_test = x[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    if verbose:
        plt.scatter(x_train, y_train, c='r', s=1., label='Train')
        plt.scatter(x_test, y_test, c='b', s=1., label='Test')
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    return df, x_train, x_test, y_train, y_test, mean, std


def get_kernel(kernel_path: str) -> Kernel:

    # Use importlib to import the script as a module
    spec = importlib.util.spec_from_file_location("kernel", kernel_path)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)

    # Access the get_gpr_kernel function and call it to retrieve the kernel
    kernel = getattr(kernel_module, "kernel")  # Get the function

    return kernel


def optimize_with_hms(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metric_fn: Callable
):
    """
    TODO This is a mock version of the function. The function should:
    1. Optimize the kernel parameters with the HMS algorithm. The optimisation metric is the `metric_fn`
    2. define the sklearn kernel with fixed bounds, to prevent kernel parameters from changing
    3. Fit the model with the new kernel (fitting only to data, without kernel hyperparameters tuning - becouse we fixed them)
    4. Return the GaussianProcessRegressor object with the new, optimized kernel
    """

    suggested_kernel = gpr.kernel_

    # TODO Optimize the kernel hyperparameters with the HMS algorithm

    # TODO Define the new kernel with fixed bounds
    fixed = "fixed"

    # Term responsible for the noise in data
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=fixed)

    # Term responsible for the nonlinear trend in data
    k1 = ConstantKernel(constant_value=10, constant_value_bounds=fixed) * \
    RBF(length_scale=500, length_scale_bounds=fixed)

    # Term responsible for the seasonal component in data
    k2 = ConstantKernel(constant_value=1, constant_value_bounds=fixed) * \
    ExpSineSquared(length_scale=1.0, periodicity=10, length_scale_bounds=fixed, periodicity_bounds=fixed)

    optimized_kernel  = k0 + k1 + k2

    # Fit the model with the new kernel
    gpr.kernel_ = optimized_kernel
    gpr.fit(x_train, y_train)

    return gpr


def plot_samples(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mean: float,
    std: float,
    n_samples: int = 10,
    normalize: bool = False,
    title: str = 'Gaussian Process',
) -> None:
    gp_prior_samples = gpr.sample_y(X=x_train, n_samples=n_samples)

    fig, ax = plt.subplots()
    for i in range(n_samples):
        sns.lineplot(x=x_train[...,0], y = gp_prior_samples[:, i], color=sns_c[1], alpha=0.2, ax=ax)
    if normalize:
        sns.lineplot(x=x_train[...,0], y=(y_train[..., 0] - mean) / std, color=sns_c[0], label='y', ax=ax) 
    else:
        sns.lineplot(x=x_train[...,0], y=y_train[..., 0], color=sns_c[0], label='y', ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(title=title, xlabel='t');
    plt.show()


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
    df['t'] = df.index
    df['y_mean'] = y_mean
    df['y_std'] = y_std
    df['y_lwr'] = df['y_mean'] - 2*df['y_std']
    df['y_upr'] = df['y_mean'] + 2*df['y_std']

    # Plot the predictions
    fig, ax = plt.subplots()
    ax.fill_between(
        x=df['t'], 
        y1=df['y_lwr'], 
        y2=df['y_upr'], 
        color=sns_c[2], 
        alpha=0.15, 
        label='+- 2*std'
    )
    sns.lineplot(x='t', y='CO2', data=df, color=sns_c[0], label = 'y_true', ax=ax)
    sns.lineplot(x='t', y='y_mean', data=df, color=sns_c[2], label='y_hat_mean', ax=ax)
    ax.axvline(x_train.shape[0], color=sns_c[3], linestyle='--', label='train-test split')
    ax.legend()
    ax.set(title=f'Test R2 score: {r2_score_test:.3f}', xlabel='t', ylabel='');
    plt.savefig(os.path.join(results_dir, 'predictions.pdf'), bbox_inches='tight')

    # Calculate prediction errors
    errors = gpr.predict(x_test) - np.squeeze(y_test)
    errors = errors.flatten()
    errors_mean = errors.mean()
    errors_std = errors.std()

    # Plot prediction errors
    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) 
    sns.regplot(x=y_test.flatten(), y=gpr.predict(x_test).flatten(), ax=ax[0])
    ax[0].plot(y_test.flatten(), y_test.flatten(), color=sns_c[3], linestyle='--', label='y=x')
    sns.histplot(data=errors, ax=ax[1], bins=25)
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
    ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
    ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
    ax[0].legend()
    ax[1].legend()
    ax[0].set(title='Test vs Predictions (Test Set)', xlabel='y_test', ylabel='y_pred');
    ax[1].set(title='Errors', xlabel='error', ylabel=None);
    plt.savefig(os.path.join(results_dir, 'errors.pdf'), bbox_inches='tight')
    
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
            title='Gaussian Process Prior Samples'
        )

    # Fit the model
    if verbose:
        print("[info] Fitting the model...")
    if optimizer == "sklearn":
        gpr.fit(x_train, y_train)
    if optimizer == "hms":
        gpr = optimize_with_hms(gpr, x_train, y_train)
    
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
            title='Gaussian Process Posterior Samples'
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
    print(f"Train R2 score: {r2_score_train:.3f}")
    print(f"Test R2 score:  {r2_score_test:.3f}")


if __name__ == '__main__':

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
