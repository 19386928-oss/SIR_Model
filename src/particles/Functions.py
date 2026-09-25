import numpy as np
import arviz as az
import pandas as pd
import statistics

# Summary statistics of posterior distribution
def summary(param_1, param_2, posterior_1, posterior_2, decimal):
    summary = {'Parameter': [param_1, param_2],
           'Mean': [az.mean(posterior_1, round_to=decimal), az.mean(posterior_2, round_to=decimal)],
           'Bias': [(0.45 - az.mean(posterior_1, round_to=decimal)), (0.08 - az.mean(posterior_2, round_to=decimal))],
           'Variance': [round(statistics.variance(posterior_1.flatten()),decimal), round(statistics.variance(posterior_2.flatten()),decimal)],
           'Std': [az.std(posterior_1, round_to=decimal), az.std(posterior_2, round_to=decimal)],
           'eti95_lb': [round(az.eti(posterior_1.flatten())[0],decimal), round(az.eti(posterior_2.flatten())[0],decimal)],
           'eti95_ub': [round(az.eti(posterior_1.flatten())[1],decimal), round(az.eti(posterior_2.flatten())[1],decimal)],
           'hdi95_lb': [round(az.hdi(posterior_1.flatten())[0],decimal), round(az.hdi(posterior_2.flatten())[0],decimal)],
           'hdi95_ub': [round(az.hdi(posterior_1.flatten())[1],decimal), round(az.hdi(posterior_2.flatten())[1],decimal)],
           'ess_bulk': [round(az.ess(posterior_1, method='bulk').item(),decimal), round(az.ess(posterior_2, method='bulk').item(),decimal)],
           'ess_tail': [round(az.ess(posterior_1, method='tail', prob=0.95).item(),decimal), round(az.ess(posterior_2, method='tail', prob=0.95).item(),decimal)],
           'rhat': [round(az.rhat(posterior_1).item(),decimal), round(az.rhat(posterior_2).item(),decimal)],
           'mcse_mean': [round(az.mcse(posterior_1, method='mean').item(),decimal), round(az.mcse(posterior_2, method='mean').item(),decimal)],
           'mcse_std': [round(az.mcse(posterior_1, method='sd').item(),decimal), round(az.mcse(posterior_2, method='sd').item(),decimal)]
           }
    df_summary = pd.DataFrame(summary)
    return display(df_summary)

def weighted_ess(chain, generation, smc):
    weights = smc[chain].weights[generation]
    w = weights / np.sum(weights)
    ess = 1 / np.sum(w**2)
    return ess