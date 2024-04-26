from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from utils import *

OPTIMAL_KERNEL_PATH = "/Users/wciezobka/agh/SemestrX/stochastyczne/MetodyStochastyczne2024/results/20240424-161901/kernel_fitted.py"


def mock_optimization_fn(initial_parameters: List[Tuple]) -> List[Tuple]:
    """
    Mock optimization function that returns the optimal kernel for CO2 data
    """
    optimal_kernel = get_kernel(kernel_path=OPTIMAL_KERNEL_PATH)
    optimal_gpr = GaussianProcessRegressor(kernel=optimal_kernel)
    optimal_kernel_params = parse_parameters(optimal_gpr)

    return optimal_kernel_params

def mock_optimization(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metric_fn: Callable = None
) -> GaussianProcessRegressor:
    """
    TODO This is a mock version of the function. The optimization function should:
    1. Extract hyperparameters and bounds from the kernel
    2. Optimize the kernel parameters with the HMS algorithm. The optimisation metric is the `metric_fn`
    3. Define the sklearn kernel with optimized hyperparameters and fixed bounds, to prevent kernel parameters from changing
    4. Fit the GPR model with the new fixed kernel to training data (fitting only to data, without kernel hyperparameters tuning - becouse we fixed them)
    5. Return the GaussianProcessRegressor object with the new, optimized kernel
    """

    # 1. Extract hyperparameters and bounds from the kernel
    initial_parameters = parse_parameters(gpr)

    # === This is The only step that should be replaced with the real optimization algorithm ===
    # 2. Optimize the kernel parameters with the mock optimization
    # - that is switch the kernel with the fitted one
    optimal_kernel_params = mock_optimization_fn(initial_parameters)
    # ===========================================================================================

    # 3. Define the sklearn kernel with optimized hyperparameters and fixed bounds
    optimal_gpr = set_parameters(gpr, optimal_kernel_params, fixed=True)

    # 4. Fit the GPR model with the new fixed kernel to training data
    optimal_gpr.fit(x_train, y_train)

    # 5. Return the GaussianProcessRegressor object with the new, optimized kernel
    return optimal_gpr