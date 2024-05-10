from typing import Callable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from gpr_evaluation import GPREvaluation

from .sea import SEA


def sea_optimization(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metric_fn: Callable,
    maximize_metric
):
    gpr_evaluation = GPREvaluation(gpr, x_train, y_train, 0.8, metric_fn, maximize_metric)

    sea = SEA(gpr_evaluation.evaluate_model,
              gpr_evaluation.get_bounds(),
              maximize_metric,
              population_size=25,
              n_generations=25,
              mutation_rate=1,
              tournament_size=3)

    optimized_params = sea.optimize()

    gpr_evaluation.set_params(optimized_params)

    return gpr_evaluation.gpr