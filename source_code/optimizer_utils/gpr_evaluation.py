from typing import Callable

from sklearn.gaussian_process import GaussianProcessRegressor
from utils import parse_parameters, fix_kernel, train_test_split, FITNESS_MAPPING
import numpy as np


class GPREvaluation:
    def __init__(self, gpr: GaussianProcessRegressor, x_train, y_train, metric_fn: str):
        self.gpr = gpr
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train, 0.8)
        self.metric_fn, self.maximize = FITNESS_MAPPING[metric_fn]
        self.params = parse_parameters(gpr)

        self.gpr.kernel = fix_kernel(self.gpr.kernel)

    def get_bounds(self):
        return np.array([bounds for name, v, bounds in self.params])

    def set_params(self, params):
        parameters = {}
        for param, value in zip(self.params, params):
            parameters[param[0]] = value
        self.gpr.kernel.set_params(**parameters)

    def evaluate_model(self, params):
        self.set_params(params)

        self.gpr.fit(self.x_train, self.y_train)
        y_pred = self.gpr.predict(self.x_test)
        metric = self.metric_fn(self.y_test, y_pred)

        return metric
