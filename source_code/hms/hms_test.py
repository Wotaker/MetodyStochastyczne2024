import numpy as np
import os
from pyhms import hms, EALevelConfig, Problem, DontStop, MetaepochLimit, SEA, get_NBC_sprout
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import mean_squared_error
from source_code.utils import *


class GPRProblem(Problem):
    def __init__(self, x_train, y_train, x_test, y_test, bounds):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self._bounds = np.array(bounds)
        self._maximize = False  # Since we are minimizing MSE

    def evaluate(self, params):
        return evaluate_model(params, self.x_train, self.y_train, self.x_test, self.y_test)

    def worse_than(self, current, candidate):
        return candidate > current

    @property
    def bounds(self):
        return self._bounds

    @property
    def maximize(self):
        return self._maximize


def evaluate_model(kernel_params, x_train, y_train, x_test, y_test):
    k0 = WhiteKernel(noise_level=kernel_params[0], noise_level_bounds="fixed")
    k1 = ConstantKernel(constant_value=kernel_params[1], constant_value_bounds="fixed") * \
         RBF(length_scale=kernel_params[2], length_scale_bounds="fixed")
    k2 = ConstantKernel(constant_value=1, constant_value_bounds="fixed") * \
         ExpSineSquared(length_scale=1.0, periodicity=kernel_params[3], length_scale_bounds="fixed",
                        periodicity_bounds="fixed")
    kernel = k0 + k1 + k2
    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return mse


# # Load your data
# df, x_train, x_test, y_train, y_test, mean, std = load_data(
#     os.path.join(os.path.dirname(__file__), '../data/co2_clean.csv'), 'CO2')
#
# gpr_problem = GPRProblem(
#     x_train=x_train,
#     y_train=y_train,
#     x_test=x_test,
#     y_test=y_test,
#     bounds=[(0.01, 0.25), (1, 500), (1, 1e4), (8, 15)]
# )
#
# # Configuration for the HMS algorithm
# config = [
#     EALevelConfig(
#         ea_class=SEA,
#         generations=2,
#         problem=gpr_problem,
#         pop_size=20,
#         mutation_std=1.0,
#         lsc=DontStop(),
#     ),
#     EALevelConfig(
#         ea_class=SEA,
#         generations=4,
#         problem=gpr_problem,
#         pop_size=10,
#         mutation_std=0.25,
#         sample_std_dev=1.0,
#         lsc=DontStop(),
#     ),
# ]
# global_stop_condition = MetaepochLimit(limit=10)
# sprout_condition = get_NBC_sprout(level_limit=4)
# hms_tree = hms(config, global_stop_condition, sprout_condition)
#
# print(hms_tree.summary())

k0 = WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
k1 = ConstantKernel(constant_value=1, constant_value_bounds="fixed") * \
     RBF(length_scale=1, length_scale_bounds="fixed")
k2 = ConstantKernel(constant_value=1, constant_value_bounds="fixed") * \
     ExpSineSquared(length_scale=1.0, periodicity=8, length_scale_bounds="fixed",
                    periodicity_bounds="fixed")
kernel = k0 + k1 + k2
model = GaussianProcessRegressor(kernel=kernel, random_state=0)

print(model.kernel.hyperparameters)
print(model.kernel.theta)
print(model.get_params())
print(model.kernel.get_params())
