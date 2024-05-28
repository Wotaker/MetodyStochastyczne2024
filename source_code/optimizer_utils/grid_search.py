from typing import Iterable, Any
from itertools import product


def _grid_parameters(param_grid: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))


class GridSearch:
    def __init__(self, optimizer, param_grid: dict, validation_count, verbose=False):
        self.optimizer = optimizer
        self.param_grid = param_grid
        self.validation_count = validation_count

        self.verbose = verbose

        self.scores = []

    def fit(self):
        for i, settings in enumerate(_grid_parameters(self.param_grid)):
            if self.verbose:
                print("Round:", i)
            self.optimizer.set_params(settings)
            _, score = self.optimizer.optimize()
            s = (settings, score)
            self.scores.append(s)
