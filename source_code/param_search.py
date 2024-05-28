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
        'population_size': [25],
        'mutation_rate': [1],
        'tournament_size': [3],
        'n_generations': [5, 10]
    }

    if optimizer == "sea":
        gpr_evaluation = GPREvaluation(gpr, x_train, y_train, mean_squared_error, maximize=False)
        sea = SEA(gpr_evaluation.evaluate_model,
                  gpr_evaluation.get_bounds(),
                  gpr_evaluation.maximize,
                  verbose=verbose
                  )
        grid_search = GridSearch(sea, param_grid, verbose)
        grid_search.fit()
        print(grid_search.scores)


if __name__ == '__main__':
    param_search("../data/co2_clean.csv", "CO2", 0.8, "../kernels/sample_kernel.py", "sea", verbose=True)
