import particles
from sir_PMCMC import ChainBinomialModel
from sir_PMCMC import ScalarStructDist
from SIR_Model import y
import time
import pickle
import multiprocess as mp
import numpy as np
import os

from particles import distributions as dists
from particles import mcmc


# Prior distributions for Parameters
prior_dict = {'beta': dists.Gamma(a = 4.5, b = 10),
              'gamma': dists.Gamma(a = 8, b = 100)}  # Uniform distribution because the prior is uninformative

my_prior = ScalarStructDist(prior_dict)      # need this because 'dict' object has no attribute 'dtype' but StructDist does

# PMMH algorithm
def pmmh_multi(seed):
    np.random.seed(seed)
    pmmh_multi = mcmc.PMMH(ssm_cls= ChainBinomialModel, 
                     prior = my_prior, 
                     data=y, 
                     Nx=1000, 
                     niter=5000, 
                     adaptive=True,
                     smc_options= {'resampling': 'systematic', 'verbose': 'True'}
                    )
    pmmh_multi.run()
    return pmmh_multi

start = time.time()
with mp.Pool() as pool:
    pool_pmcmc = pool.map(pmmh_multi, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Save PMMH object
with open('pmmh_test_cpu.pkl', 'wb') as output_multi:
    pickle.dump(pool_pmcmc, output_multi)

print("Chain saved.")
print(os.path.abspath("pmmh_test_cpu.pkl"))