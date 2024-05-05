from typing import Callable
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from pyhms import hms, EALevelConfig, Problem, DontStop, MetaepochLimit, SEA, get_NBC_sprout
from sklearn.metrics import r2_score
from utils import fix_kernel, parse_parameters


def create_kernel(noise_level, constant_value, length_scale, periodicity, bounds='fixed'):
    k0 = WhiteKernel(noise_level ** 2, noise_level_bounds=bounds)
    k1 = ConstantKernel(constant_value, constant_value_bounds=bounds) * \
         RBF(length_scale, length_scale_bounds=bounds)
    k2 = ConstantKernel(constant_value=1, constant_value_bounds=bounds) * \
         ExpSineSquared(length_scale=1.0, periodicity=periodicity, length_scale_bounds=bounds,
                        periodicity_bounds=bounds)
    return fix_kernel(k0 + k1 + k2)


class GPRProblem(Problem):
    def __init__(self, gpr: GaussianProcessRegressor, x_train, y_train, bounds, metric_fn):
        self.gpr = gpr
        self.x_train = x_train
        self.y_train = y_train
        self._bounds = np.array(bounds)
        self._maximize = True
        self.metric_fn = metric_fn
        self.params = parse_parameters(gpr)

        self.gpr.kernel = fix_kernel(self.gpr.kernel)

    def evaluate(self, params):
        return self.evaluate_model(params)

    def evaluate_model(self, params):
        parameters = {}
        for param, value in zip(self.params, params):
            parameters[param[0]] = value
        self.gpr.kernel.set_params(**parameters)

        self.gpr.fit(self.x_train, self.y_train)
        y_pred = self.gpr.predict(self.x_train)
        metric = self.metric_fn(self.y_train, y_pred)

        return metric

    def worse_than(self, current, candidate):
        return candidate > current

    @property
    def bounds(self):
        return self._bounds

    @property
    def maximize(self):
        return self._maximize


def hms_optimization(gpr, x_train, y_train, metric_fn=r2_score):
    initial_bounds = [(0.01, 0.25), (1, 500), (1, 1e4), (8, 15)]

    gpr_problem = GPRProblem(gpr, x_train, y_train, initial_bounds, metric_fn)

    config = [
        EALevelConfig(ea_class=SEA, generations=2, problem=gpr_problem, pop_size=20, mutation_std=1.0, lsc=DontStop()),
        EALevelConfig(ea_class=SEA, generations=4, problem=gpr_problem, pop_size=10, mutation_std=0.25,
                      sample_std_dev=1.0, lsc=DontStop()),
    ]
    global_stop_condition = MetaepochLimit(limit=10)
    sprout_condition = get_NBC_sprout(level_limit=4)
    hms_tree = hms(config, global_stop_condition, sprout_condition)

    best_individual = hms_tree.best_individual
    param_values = best_individual.genome

    optimized_kernel = create_kernel(*param_values)

    gpr.kernel_ = optimized_kernel
    gpr.fit(x_train, y_train)

    return gpr
