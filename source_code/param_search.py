from sklearn.metrics import mean_squared_error, r2_score

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
        verbose=verbose
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
        'mutation_rate': [0.25, 1],
        'tournament_size': [3],
        'n_generations': [25]
    }

    if optimizer == "sea":
        gpr_evaluation = GPREvaluation(gpr, x_train, y_train, 0.8, mean_squared_error, maximize_metric=False)
        sea = SEA(gpr_evaluation.evaluate_model,
                  gpr_evaluation.get_bounds(),
                  gpr_evaluation.maximize_metric,
                  )
        grid_search = GridSearch(sea, param_grid, 1, True)
        grid_search.fit()
        print(grid_search.scores)

if __name__ == '__main__':
    param_search("../data/co2_clean.csv", "CO2", 0.8, "../kernels/sample_kernel.py", "sea")