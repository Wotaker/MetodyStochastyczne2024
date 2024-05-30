from sea.sea import SEA
from optimizer_utils.gpr_evaluation import GPREvaluation
from optimizer_utils.grid_search import GridSearch
from utils import *


def param_search(
        data_path: str,
        column: str,
        split: float,
        kernel_path: str,
        optimizer: str,
        metrics: str,
        verbose: bool = False
):
    df, x_train, x_test, y_train, y_test, mean, std = load_data(
        data_path=data_path,
        column=column,
        split=split,
        verbose=False
    )

    kernel = get_kernel(kernel_path=kernel_path)

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        alpha=0.0,
    )

    param_grid = {
        'population_size': [10, 20],
        'mutation_rate': [0.1, 0.5, 1],
        'tournament_size': [3],
        'n_generations': [5],
    }

    iterations = 3

    if optimizer == "sea":
        gpr_evaluation = GPREvaluation(gpr, x_train, y_train, metrics)
        sea = SEA(gpr_evaluation.evaluate_model,
                  gpr_evaluation.get_bounds(),
                  gpr_evaluation.maximize,
                  verbose=verbose
                  )
        grid_search = GridSearch(sea, gpr_evaluation, param_grid, iterations, verbose)
        grid_search.fit()


# Change "mse" to "r2" id you want to switch metric. The evaluation and params comparison should be adjusted accordingly
if __name__ == '__main__':
    param_search("./data/co2_clean.csv", "CO2", 0.8, "./kernels/sample_kernel.py", "sea", "r2", verbose=True)
