from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from ..utils import parse_parameters, fix_kernel

from pyhms import minimize


def hms_optimization(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metric_fn: Callable
):
    """
    TODO This is a mock version of the function. The function should:
    # 1. Extract hyperparameters and bounds from the kernel
    # 2. Optimize the kernel parameters with the HMS algorithm. The optimisation metric is the `metric_fn`
    # 3. Define the sklearn kernel with optimized hyperparameters and fixed bounds, to prevent kernel parameters from changing
    # 4. Fit the GPR model with the new fixed kernel to training data (fitting only to data, without kernel hyperparameters tuning - becouse we fixed them)
    5. Return the GaussianProcessRegressor object with the new, optimized kernel
    """
    gpr_evaluation = GPREvaluation(gpr, x_train, y_train, metric_fn)

    solution = minimize(
        fun=gpr_evaluation.evaluate_model,
        bounds=gpr_evaluation.get_bounds(),
        maxfun=10000,
        log_level="debug"
    )

    # TODO Define the new kernel with fixed bounds

    return gpr


class GPREvaluation:
    # TODO move this class to a separate file
    def __init__(self, gpr: GaussianProcessRegressor, x_train, y_train, metric_fn: Callable):
        self.gpr = gpr
        self.x_train = x_train
        self.y_train = y_train
        self.metric_fn = metric_fn
        self.params = parse_parameters(gpr)

        self.gpr.kernel = fix_kernel(self.gpr.kernel)

    def get_bounds(self):
        return np.array([bounds for name, bounds in self.params])

    def evaluate_model(self, params):
        parameters = {}
        for param, value in zip(self.params, params):
            parameters[param[0]] = value
        self.gpr.kernel.set_params(**parameters)

        self.gpr.fit(self.x_train, self.y_train)
        y_pred = self.gpr.predict(self.x_train)
        metric = self.metric_fn(self.y_train, y_pred)

        return metric
