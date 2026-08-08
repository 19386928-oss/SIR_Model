import particles
from sir_PMCMC import ChainBinomialModel
from sir_PMCMC import ScalarStructDist
from SIR_Model import y
import time
import pickle
import os

from particles import distributions as dists
from particles import mcmc


# Prior distributions for Parameters
prior_dict = {'beta': dists.Uniform(a=0, b=0.5),
              'gamma': dists.Uniform(a= 0, b= 0.5)}  # Uniform distribution because the prior is uninformative

my_prior = ScalarStructDist(prior_dict)      # need this because 'dict' object has no attribute 'dtype' but StructDist does

# PMMH algorithm
start = time.time()
pmmh = mcmc.PMMH(ssm_cls= ChainBinomialModel, prior = my_prior, data=y, Nx=10000, niter=5000, adaptive=True)
pmmh.run()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Save PMMH object

with open("pmmh_chain.pkl", 'wb') as output:
    pickle.dump(pmmh.chain, output)

print("Chain saved.")
print(os.path.getsize("pmmh_chain.pkl"))
print(os.getcwd())
print(os.path.abspath("pmmh_chain.pkl"))
if os.path.exists("pmmh_chain.pkl"):
    print(os.path.getsize("pmmh_chain.pkl"))
