from typing import List, Callable

import random
import numpy as np
from tqdm import tqdm


class SEA:
    def __init__(
            self,
            fun: Callable,
            bounds: List,
            maximize: bool,
            population_size: int,
            n_generations: int,
            mutation_rate: float,
            tournament_size: int,
            verbose: bool = False
        ):
        self.fun = fun
        self.bounds = bounds
        self.maximize = maximize
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.verbose = verbose

    def _initialize_population(self):
        return [[np.random.uniform(*param_bounds) for param_bounds in self.bounds] for _ in range(self.population_size)]

    def _tournament_selection(self, population, scores):
        tournament = random.sample(list(zip(population, scores)), self.tournament_size)
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0]

    def _crossover(self, parent1, parent2):
        child = [random.choice(pair) for pair in zip(parent1, parent2)]
        return child

    def _mutate(self, child):
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(child) - 1)
            child[index] *= random.uniform(0.9, 1.1)
        return child

    def optimize(self):
        population = self._initialize_population()
        if not self.maximize:
            best_score = float('inf')
        else:
            best_score = float('-inf')
        best_params = None

        for _ in tqdm(range(self.n_generations), disable=not self.verbose):
            scores = [self.fun(individual) for individual in population]
            if not self.maximize:
                best_current = min(scores)
                if best_current < best_score:
                    best_score = best_current
                    best_params = population[scores.index(best_current)]
            else:
                best_current = max(scores)
                if best_current > best_score:
                    best_score = best_current
                    best_params = population[scores.index(best_current)]

            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)
                while np.array_equal(parent1, parent2):
                    parent2 = self._tournament_selection(population, scores)

                child1 = self._mutate(self._crossover(parent1, parent2))
                child2 = self._mutate(self._crossover(parent1, parent2))
                new_population.extend([child1, child2])

            population = new_population

        return best_params
