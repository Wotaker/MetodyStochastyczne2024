from typing import Tuple, List

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from datetime import datetime
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor


def create_experiment_dir(dir_path: str, kernel_path: str) -> str:

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create directory
    dir_path = os.path.join(dir_path, timestamp)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Copy configuration file to experiment directory
    os.system(f"cp {kernel_path} {os.path.join(dir_path, 'kernel_initial.py')}")
    
    # Return
    return dir_path


def train_test_split(x, y, train_size: float):
    n = len(x)
    n_train = int(n * train_size)

    x_train = x[:n_train]
    x_test = x[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return x_train, x_test, y_train, y_test


def load_data(data_path: str, column: str, split: float = 0.8, verbose: bool = False) -> Tuple:

    # Load the data
    df = pd.read_csv(data_path)
    mean = df[column].mean()
    std = df[column].std()

    # Create training and testing sets
    n = len(df)

    # Convert data to numpy arrays
    x = df.index.values.reshape(-1, 1)
    y = df[column].values.reshape(n, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, split)

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


def fix_kernel(kernel: Kernel) -> Kernel:
    kernel_params_dict = kernel.get_params()
    bounds_keys = [key for key in kernel_params_dict.keys() if 'bounds' in key]
    for key in bounds_keys:
        kernel_params_dict[key] = "fixed"
    kernel.set_params(**kernel_params_dict)
    return kernel


def parse_parameters(gpr: GaussianProcessRegressor) -> List[Tuple]:
    """
    Returns a list of kernel parameters with their values and bounds. The names of parameters are accepted by the
    kernel.setparams(**{name:value}) method.
    """

    hyperparameters = gpr.kernel.hyperparameters
    theta = gpr.kernel.theta
    kernel_parameters = []
    for h, v in zip(hyperparameters, theta):
        kernel_parameters.append((h.name, np.exp(v), tuple(*h.bounds)))

    return kernel_parameters


def set_parameters(gpr: GaussianProcessRegressor, kernel_parameters: List[Tuple], fixed: bool = True) -> GaussianProcessRegressor:
    """
    Sets the kernel parameters of a GaussianProcessRegressor object.
    """

    kernel = gpr.kernel
    kernel.set_params(**{kernel_parameters[i][0]: kernel_parameters[i][1] for i in range(len(kernel_parameters))})
    gpr.kernel = fix_kernel(kernel) if fixed else kernel

    return gpr
