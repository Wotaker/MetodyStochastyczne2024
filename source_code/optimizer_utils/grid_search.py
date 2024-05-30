from typing import Iterable, Any
from itertools import product
import numpy as np
import pandas as pd
import os
from datetime import datetime


def _grid_parameters(param_grid: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))


class GridSearch:
    def __init__(self, optimizer, evaluator, param_grid: dict, iters: int, verbose=False):
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.param_grid = param_grid
        self.iters = iters
        self.combinations = np.prod([len(v) for v in self.param_grid.values()])
        self.verbose = verbose

    def fit(self):
        all_scores = []
        for i, settings in enumerate(_grid_parameters(self.param_grid)):
            if self.verbose:
                print(f"\nConbination: {i + 1}/{self.combinations}")
                print(f"Settings: {settings}")
            scores = []
            for _ in range(self.iters):
                print(f"Iteration: {_ + 1}/{self.iters}")
                self.optimizer.set_params(settings)
                _, score = self.optimizer.optimize()
                scores.append(score)
                print(f"Scores: {score}")
            s = settings.copy()
            s['iterations'] = self.iters
            s['metric'] = self.evaluator.metric_fn.__name__
            s['mean_score'] = np.mean(scores)
            s['best_score'] = np.min(scores) if not self.optimizer.maximize else np.max(scores)
            all_scores.append(s)
            print(f"SUMMARY: Settings: {s}, Scores: {scores}")

        df = pd.DataFrame(all_scores)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        optimizer_name = self.optimizer.__class__.__name__
        metric = self.evaluator.metric_fn.__name__
        dir_path = os.path.join("results", f"{optimizer_name}_{metric}_{timestamp}")
        os.makedirs(dir_path, exist_ok=True)
        df.to_csv(os.path.join(dir_path, 'grid_search_results.csv'), index=False)
