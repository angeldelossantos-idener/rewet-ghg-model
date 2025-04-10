from scipy.optimize import differential_evolution

import numpy as np
# import pandas as pd


def ghg_model(x):
    "Not even close"
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

bounds = [(-5, 5), (-5, 5)]
result = differential_evolution(ghg_model, bounds, rng=1)
print(result.x)
print(result.fun)