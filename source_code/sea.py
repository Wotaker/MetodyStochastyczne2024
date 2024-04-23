import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import mean_squared_error
import random
import os

def evaluate_model(kernel_params, X_train, y_train, X_test, y_test):
    k0 = WhiteKernel(noise_level=kernel_params[0]**2)
    k1 = ConstantKernel(constant_value=kernel_params[1]) * \
         RBF(length_scale=kernel_params[2])
    k2 = ConstantKernel(constant_value=1) * \
         ExpSineSquared(length_scale=1.0, periodicity=kernel_params[3])
    kernel = k0 + k1 + k2
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def mutate(kernel_params, mutation_rate=0.1):
    if random.random() < mutation_rate:
        param_index = random.randint(0, len(kernel_params) - 1)
        kernel_params[param_index] *= random.uniform(0.5, 1.5)
    return kernel_params

def crossover(parent1, parent2):
    child = [random.choice(pair) for pair in zip(parent1, parent2)]
    return child

def genetic_algorithm(X_train, y_train, X_test, y_test, population_size, n_generations):
    population = [np.random.rand(4) for _ in range(population_size)] 
    best_score = float('inf')
    best_params = None
    
    for _ in range(n_generations):
        scores = [evaluate_model(ind, X_train, y_train, X_test, y_test) for ind in population]
        best_current = min(scores)
        if best_current < best_score:
            best_score = best_current
            best_params = population[scores.index(best_current)]
        
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent1, parent2))
            new_population.extend([child1, child2])
        population = new_population

    return best_params, best_score


# Load data
df = pd.read_csv(os.path.join(os.path.dirname(__file__),'../data/CSCO_data.csv'))
df = df[:500]
split_ratio = 0.8
n = len(df)
n_train = int(n * split_ratio)

# Convert data to numpy arrays
x = df.index.values.reshape(-1, 1)
y = df['Close'].values.reshape(n, 1)

# Split data into training and test sets
X_train = x[:n_train]
X_test = x[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]


population_size = 10
n_generations = 10
best_params, best_score = genetic_algorithm(X_train, y_train, X_test, y_test, population_size, n_generations)
print("Najlepsze parametry:", best_params)
print("Najmniejszy MSE:", best_score)

