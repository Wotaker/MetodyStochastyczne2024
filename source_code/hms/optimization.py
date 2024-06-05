from typing import Callable
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from pyhms import hms, EALevelConfig, Problem, DontStop, MetaepochLimit, SEA, get_NBC_sprout
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import fix_kernel


def create_kernel(noise_level, constant_value, length_scale, periodicity, bounds='fixed'):
    k0 = WhiteKernel(noise_level ** 2, noise_level_bounds=bounds)
    k1 = ConstantKernel(constant_value, constant_value_bounds=bounds) * \
         RBF(length_scale, length_scale_bounds=bounds)
    k2 = ConstantKernel(constant_value=1, constant_value_bounds=bounds) * \
         ExpSineSquared(length_scale=1.0, periodicity=periodicity, length_scale_bounds=bounds,
                        periodicity_bounds=bounds)
    return fix_kernel(k0 + k1 + k2)


def evaluate_model(kernel_params, x_train, y_train, x_test, y_test, metric_fn):
    kernel = create_kernel(*kernel_params)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(x_train, y_train)
    predictions = gpr.predict(x_test)
    return metric_fn(y_test, predictions)


class GPRProblem(Problem):
    def __init__(self, x_train, y_train, x_test, y_test, bounds):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self._bounds = np.array(bounds)

        self.metric_fn = r2_score  # ew. r2_score i True
        self._maximize = True

    def evaluate(self, params):
        return evaluate_model(params, self.x_train, self.y_train, self.x_test, self.y_test, self.metric_fn)

    def worse_than(self, current, candidate):
        return candidate > current

    @property
    def bounds(self):
        return self._bounds

    @property
    def maximize(self):
        return self._maximize


def hms_optimization(x_train, y_train, x_test, y_test):
    initial_bounds = [(0.01, 0.25), (1, 500), (1, 1e4), (8, 15)]

    gpr_problem = GPRProblem(x_train, y_train, x_test, y_test, initial_bounds)

    config = [
        EALevelConfig(ea_class=SEA, generations=10, problem=gpr_problem, pop_size=20, mutation_std=0.2, lsc=DontStop()),
        EALevelConfig(ea_class=SEA, generations=20, problem=gpr_problem, pop_size=10, mutation_std=0.05,
                      sample_std_dev=1.0, lsc=DontStop()),
    ]
    global_stop_condition = MetaepochLimit(limit=10)
    sprout_condition = get_NBC_sprout(level_limit=4)
    hms_tree = hms(config, global_stop_condition, sprout_condition)

    best_individual = hms_tree.best_individual
    param_values = best_individual.genome

    optimized_kernel = create_kernel(*param_values)

    optimized_gpr = GaussianProcessRegressor(kernel=optimized_kernel, random_state=0)
    optimized_gpr.fit(x_train, y_train)

    return optimized_gpr
