# Simulate discrete time markov chain SIR epidemic model in Particles

import particles
import numpy as np
from scipy import stats 
from scipy.stats import binom
from particles import distributions as dists
from particles import state_space_models as ssm
from collections import OrderedDict

# class DiscreteDist copied from Particles documentation, no changes made except dim is 2 for multivariate
class DiscreteDist:
    dim = 2     # multivariate distribution (default is 1 for univariate)
    dtype = np.int64

    def shape(self, size):
        if size is None:
            return None
        else:
            return (size,) if self.dim == 1 else (size, self.dim)
        
    def logpdf(self, x):
        raise NotImplementedError
    
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    
    def rvs(self, size=None):
        raise NotImplementedError
    
    def ppf(self, u):
        raise NotImplementedError
    

# changed the order of n & p to align with scipy.stats documentation
class NegativeBinomial(DiscreteDist):
    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p
    def rvs(self, size = None):
        return np.random.negative_binomial(self.n, self.p, size=size)
    def logpdf(self,x):
        return stats.nbinom.logpmf(x, self.n, self.p) # changed the order of n & p
    def ppf(self, u):
        return stats.nbinom.ppf(u, self.n, self.p) # changed the order of n & p

# Initial state of state space model (PX0)
def Initial(N, n_i):  # where xp = X_{t-1}
    
    chainrule = OrderedDict()
    chainrule['new_inf'] = dists.Dirac(0)
    chainrule['new_rec'] = dists.Dirac(0)
    chainrule['S'] = dists.Dirac(N-n_i)
    chainrule['I'] = dists.Dirac(n_i)
    chainrule['R'] = dists.Dirac(0)
    return dists.StructDist(chainrule)

# Chain binomial process (PX)
def Binomial(xp, beta, gamma, N):  # where xp = X_{t-1}
    # probabilites for new infetions and recoveries
    infection_prob = 1 - np.exp(-beta * xp['I'] / N)
    recovery_prob = 1 - np.exp(-gamma)
    # probability = np.clip(probability, 0, 1)
    
    # Update compartments
    chainrule = OrderedDict()
    chainrule['new_inf'] = dists.Binomial(n = xp['S'].astype(int), p = infection_prob)
    chainrule['new_rec'] = dists.Binomial(n = xp['I'].astype(int), p = recovery_prob)
    chainrule['S'] = dists.Cond(lambda x: dists.Dirac(xp['S'].astype(int) - x['new_inf']))
    chainrule['I'] = dists.Cond(lambda x: dists.Dirac(xp['I'].astype(int) + x['new_inf'] - x['new_rec']))
    chainrule['R'] = dists.Cond(lambda x: dists.Dirac(xp['R'].astype(int) + x['new_rec']))
    return dists.StructDist(chainrule)

# State Space Model for noisy incidence count
class ChainBinomialModel(ssm.StateSpaceModel):
    default_params = {'N': 10000, 'n_i': 10, 'rho': .25, 'phi': 5}
    def PX0(self):                                                      # Initial state of SIR
        return Initial(self.N, self.n_i)
    def PX(self, t, xp):                                                # Hidden Markov process
        return Binomial(xp, self.beta, self.gamma, self.N)
    def PY(self, t, xp, x):                                             # Observation model 
        #self.rho = rho  # reporting probability
        mu = self.rho * x['new_inf']     # mean (rho * incidences)
        mu2 = np.maximum(mu, 1e-6)
        #self.phi = phi     # dispersion parameter
        p = self.phi/(self.phi + mu2)
        return NegativeBinomial(n = self.phi, p = p)



# Change NumPy array to scalar
class ScalarStructDist(dists.StructDist):
    def logpdf(self, theta):
        out = super().logpdf(theta)
        if isinstance(out, np.ndarray) and out.shape == (1,):
            return out[0]
        return out

# Custom Moments function to account for ordered dictionary in state space model (Moments needs numerical array)
def custom_mom(W, X):
    X_arr = np.column_stack([
        X['new_inf'],
        X['new_rec'],
        X['S'],
        X['I'],
        X['R'],
    ])

    mean = np.average(X_arr, weights=W, axis=0)  
    var = np.average((X_arr - mean)**2, weights = W, axis = 0)
    return {'mean': mean, 'var': var}