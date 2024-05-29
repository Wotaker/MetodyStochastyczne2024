from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from optimizer_utils.gpr_evaluation import GPREvaluation

from .sea import SEA


def sea_optimization(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metric_fn: Callable,
    maximize: bool,
    verbose: bool
):
    gpr_evaluation = GPREvaluation(gpr, x_train, y_train, metric_fn, maximize)

    sea = SEA(
        gpr_evaluation.evaluate_model,
        gpr_evaluation.get_bounds(),
        gpr_evaluation.maximize,
        population_size=100,
        n_generations=20,
        mutation_rate=0.1,
        tournament_size=3,
        verbose=verbose
    )

    optimized_params, _ = sea.optimize()

    gpr_evaluation.set_params(optimized_params)

    return gpr_evaluation.gpr