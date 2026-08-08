import particles
from sir_PMCMC import ChainBinomialModel
from sir_PMCMC import ScalarStructDist
from sir_PMCMC import custom_mom
import time
import pickle
import pandas as pd
import numpy as np
from particles import distributions as dists
from particles import mcmc



# Simulate SIR epidemic in Particles
model = ChainBinomialModel(N = 10000, beta = 0.3, gamma = 0.1, rho = 0.25, phi = 2, n_i =10)
x, y = model.simulate(100)

simulation_df = pd.DataFrame(np.concatenate(x))
