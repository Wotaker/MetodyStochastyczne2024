import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import *
import random


# def create_kernel(noise_level, constant_value, length_scale, periodicity, bounds='fixed'):
#     k0 = WhiteKernel(noise_level=noise_level ** 2, noise_level_bounds=(0.01, 0.25))
#     k1 = ConstantKernel(constant_value=constant_value, constant_value_bounds=(1, 500)) * \
#          RBF(length_scale=length_scale, length_scale_bounds=(1, 1e4))
#     k2 = ConstantKernel(constant_value=1, constant_value_bounds="fixed") * \
#          ExpSineSquared(length_scale=1.0, periodicity=periodicity, periodicity_bounds=(8, 15))
#     return fix_kernel(k0 + k1 + k2)

def create_kernel(noise_level, constant_value, length_scale, periodicity, bounds='fixed'):
    k0 = WhiteKernel(noise_level ** 2, noise_level_bounds=bounds)
    k1 = ConstantKernel(constant_value, constant_value_bounds=bounds) * \
         RBF(length_scale, length_scale_bounds=bounds)
    k2 = ConstantKernel(constant_value=1, constant_value_bounds=bounds) * \
         ExpSineSquared(length_scale=1.0, periodicity=periodicity, length_scale_bounds=bounds,
                        periodicity_bounds=bounds)
    return fix_kernel(k0 + k1 + k2)


def evaluate_model(kernel_params, x_train, y_train, x_test, y_test, metric_fn):
    kernel = create_kernel(*kernel_params)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(x_train, y_train)
    predictions = gpr.predict(x_test)
    return metric_fn(y_test, predictions)


def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        noise_level = np.random.uniform(0.01, 0.25)
        constant_value = np.random.uniform(1, 500)
        length_scale = np.random.uniform(1, 1e4)
        periodicity = np.random.uniform(8, 15)
        population.append([noise_level, constant_value, length_scale, periodicity])
    return population


# def mutate(kernel_params, mutation_rate):
#     if random.random() < mutation_rate:
#         param_index = random.randint(0, len(kernel_params) - 1)
#         kernel_params[param_index] *= random.uniform(0.9, 1.1)
#     return kernel_params


# def mutate(kernel_params, mutation_rate):
#     initial_bounds = [(0.01, 0.25), (1, 500), (1, 1e4), (8, 15)]
#     std_devs = [np.std(bound) for bound in initial_bounds]
#     for i in range(len(kernel_params)):
#         if random.random() < mutation_rate:
#             kernel_params[i] += random.uniform(-std_devs[i], std_devs[i])
#             kernel_params[i] = max(initial_bounds[i][0], min(kernel_params[i], initial_bounds[i][1]))
#     return kernel_params

def mutate(kernel_params, mutation_rate, generation, n_generations, start_f, end_f):
    initial_bounds = [(0.01, 0.25), (1, 500), (1, 1e4), (8, 15)]
    std_devs = [np.std(bound) for bound in initial_bounds]
    mutation_range = ((end_f - start_f) / n_generations) * np.log(
        generation) + start_f
    for i in range(len(kernel_params)):
        std_devs[i] *= mutation_range
        if random.random() < mutation_rate:
            kernel_params[i] += random.uniform(-std_devs[i], std_devs[i])
            kernel_params[i] = max(initial_bounds[i][0], min(kernel_params[i], initial_bounds[i][1]))
    return kernel_params


# def crossover(parent1, parent2):
#     child = [random.choice(pair) for pair in zip(parent1, parent2)]
#     return child


# def crossover(parent1, parent2):
#     weight = random.random()
#     child = [
#         weight * gene1 + (1 - weight) * gene2
#         for gene1, gene2 in zip(parent1, parent2)
#     ]
#     return child


def crossover(parent1, parent2):
    child = [
        random.choice([gene1, gene2])
        for gene1, gene2 in zip(parent1, parent2)
    ]
    return child


# def crossover(parent1, parent2):
#     crossover_point = random.randint(1, len(parent1) - 1)
#     child = parent1[:crossover_point] + parent2[crossover_point:]
#     return child


def tournament_selection(population, scores, tournament_size):
    tournament = random.sample(list(zip(population, scores)), tournament_size)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]


# def sea_optimization(x_train, y_train, x_test, y_test, population_size=25, n_generations=25, mutation_rate=1,
#                      tournament_size=3):
#     population = initialize_population(population_size)
#     best_score = float('inf')
#     best_params = None
#
#     for _ in range(n_generations):
#         scores = [evaluate_model(ind, x_train, y_train, x_test, y_test, mean_squared_error) for ind in population]
#         best_current = min(scores)
#         if best_current < best_score:
#             best_score = best_current
#             best_params = population[scores.index(best_current)]
#
#         new_population = []
#         for _ in range(population_size // 2):
#             parent1 = tournament_selection(population, scores, tournament_size)
#             parent2 = tournament_selection(population, scores, tournament_size)
#             while np.array_equal(parent1, parent2):
#                 parent2 = tournament_selection(population, scores, tournament_size)
#             child1 = mutate(crossover(parent1, parent2), mutation_rate)
#             child2 = mutate(crossover(parent1, parent2), mutation_rate)
#             new_population.extend([child1, child2])
#         population = new_population
#
#     optimized_kernel = create_kernel(*best_params)
#     optimized_gpr = GaussianProcessRegressor(kernel=optimized_kernel, random_state=0)
#     optimized_gpr.fit(x_train, y_train)
#     return optimized_gpr

def sea_optimization(x_train, y_train, x_test, y_test, population_size=25, n_generations=25, mutation_rate=0.25,
                     tournament_size=3, elitism_rate=0.1, start_f=0.5, end_f=0.01):
    population = initialize_population(population_size)
    generation = 1
    best_score = float('inf')
    best_params = None

    for _ in range(n_generations):
        scores = [evaluate_model(ind, x_train, y_train, x_test, y_test, mean_squared_error) for ind in population]
        best_current = min(scores)
        if best_current < best_score:
            best_score = best_current
            best_params = population[scores.index(best_current)]

        elite_count = max(1, int(population_size * elitism_rate))
        elite_indices = np.argsort(scores)[:elite_count]
        elites = [population[i] for i in elite_indices]

        new_population = elites[:]
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)
            while np.array_equal(parent1, parent2):
                parent2 = tournament_selection(population, scores, tournament_size)
            child1 = mutate(crossover(parent1, parent2), mutation_rate, generation, n_generations, start_f, end_f)
            child2 = mutate(crossover(parent1, parent2), mutation_rate, generation, n_generations, start_f, end_f)
            new_population.extend([child1, child2])

        population = new_population[:population_size]
        generation += 1

    optimized_kernel = create_kernel(*best_params)
    optimized_gpr = GaussianProcessRegressor(kernel=optimized_kernel, random_state=0)
    optimized_gpr.fit(x_train, y_train)
    return optimized_gpr
