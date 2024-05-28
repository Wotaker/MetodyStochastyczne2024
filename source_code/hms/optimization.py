from typing import Callable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from optimizer_utils.gpr_evaluation import GPREvaluation

from pyhms import hms, EALevelConfig, FunctionProblem, DontStop, MetaepochLimit, SEA, get_NBC_sprout


def hms_optimization(
    gpr: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metric_fn: Callable,
    maximize: bool
):

    gpr_evaluation = GPREvaluation(gpr, x_train, y_train, 0.8, metric_fn, maximize)

    problem = FunctionProblem(gpr_evaluation.evaluate_model,
                              bounds=gpr_evaluation.get_bounds(),
                              maximize=gpr_evaluation.maximize)
    config = [
        EALevelConfig(ea_class=SEA, generations=2, problem=problem, pop_size=20, mutation_std=0.01, lsc=DontStop()),
        EALevelConfig(ea_class=SEA, generations=4, problem=problem, pop_size=10, mutation_std=0.005,
                      sample_std_dev=1.0, lsc=DontStop()),
    ]
    global_stop_condition = MetaepochLimit(limit=10)
    sprout_condition = get_NBC_sprout(level_limit=4)
    hms_tree = hms(config, global_stop_condition, sprout_condition)

    best_individual = hms_tree.best_individual
    optimized_params = best_individual.genome

    gpr_evaluation.set_params(optimized_params)

    return gpr_evaluation.gpr



