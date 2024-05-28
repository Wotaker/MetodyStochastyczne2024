from typing import Iterable, Any
from itertools import product
import numpy as np


def _grid_parameters(param_grid: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))


class GridSearch:
    def __init__(self, optimizer, param_grid: dict, verbose=False):
        self.optimizer = optimizer
        self.param_grid = param_grid

        self.iterations = np.prod([len(v) for v in self.param_grid.values()])
        self.verbose = verbose

        self.scores = []

    def fit(self):
        for i, settings in enumerate(_grid_parameters(self.param_grid)):
            if self.verbose:
                print(f"\nIteration: {i+1}/{self.iterations}")
            self.optimizer.set_params(settings)
            _, score = self.optimizer.optimize()
            s = (settings, score)
            self.scores.append(s)
