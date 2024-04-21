from sklearn.gaussian_process.kernels import *

# Term responsible for the noise in data
k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))

# Term responsible for the nonlinear trend in data
k1 = ConstantKernel(constant_value=10, constant_value_bounds=(1, 500)) * \
  RBF(length_scale=500, length_scale_bounds=(1, 1e4))

# Term responsible for the seasonal component in data
k2 = ConstantKernel(constant_value=1) * \
  ExpSineSquared(length_scale=1.0, periodicity=10, periodicity_bounds=(8, 15))

kernel = k0 + k1 + k2
