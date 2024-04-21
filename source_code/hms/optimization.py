from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *


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