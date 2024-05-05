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


def mutate(kernel_params, mutation_rate):
    if random.random() < mutation_rate:
        param_index = random.randint(0, len(kernel_params) - 1)
        kernel_params[param_index] *= random.uniform(0.9, 1.1)
    return kernel_params


def crossover(parent1, parent2):
    child = [random.choice(pair) for pair in zip(parent1, parent2)]
    return child


def tournament_selection(population, scores, tournament_size):
    tournament = random.sample(list(zip(population, scores)), tournament_size)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]


def sea_optimization(x_train, y_train, x_test, y_test, population_size=30, n_generations=30, mutation_rate=1,
                     tournament_size=3):
    population = initialize_population(population_size)
    best_score = float('inf')
    best_params = None

    for _ in range(n_generations):
        scores = [evaluate_model(ind, x_train, y_train, x_test, y_test, mean_squared_error) for ind in population]
        best_current = min(scores)
        if best_current < best_score:
            best_score = best_current
            best_score = best_current
            best_params = population[scores.index(best_current)]

        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)
            while np.array_equal(parent1, parent2):
                parent2 = tournament_selection(population, scores, tournament_size)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent1, parent2), mutation_rate)
            new_population.extend([child1, child2])
        population = new_population

    optimized_kernel = create_kernel(*best_params)
    optimized_gpr = GaussianProcessRegressor(kernel=optimized_kernel, random_state=0)
    optimized_gpr.fit(x_train, y_train)
    return optimized_gpr
