import numpy as np

# SIR simulation
def stochastic_sir(beta, gamma, N, n_i, timesteps):
    
    # Simulates a stochastic SIR model using the chain binomial process.
    
    S = N - n_i
    I = n_i
    R = 0
    S_arr, I_arr, R_arr = [S], [I], [R]
    incidence = [0]

    for _ in range(timesteps-1):

        # Infection process (Binomial draw of new infections)
        new_infected = np.random.binomial(S, 1 - np.exp(-beta * I / N))

        # Recovery process
        new_recovered = np.random.binomial(I, 1 - np.exp(-gamma))

        # Update compartments
        S -= new_infected
        I += new_infected - new_recovered
        R += new_recovered

        # Store results
        S_arr.append(S)
        I_arr.append(I)
        R_arr.append(R)
        incidence.append(new_infected)

    return {"S": S_arr, "I": I_arr, "R": R_arr, "incidence": incidence}


# SIR simulation with noisy incidences
def stochastic_sir_noisy(beta, gamma, N, n_i, timesteps, rho, phi):
    
    # Simulates a stochastic SIR model using the chain binomial process.
    
    S = N - n_i
    I = n_i
    R = 0
    S_arr, I_arr, R_arr = [S], [I], [R]
    incidence = [0]
    noisy = []

    for _ in range(timesteps-1):

        # Infection process (Binomial draw of new infections)
        new_infected = np.random.binomial(S, 1 - np.exp(-beta * I / N))

        # Recovery process
        new_recovered = np.random.binomial(I, 1 - np.exp(-gamma))

        # Update compartments
        S -= new_infected
        I += new_infected - new_recovered
        R += new_recovered

        # Store results
        S_arr.append(S)
        I_arr.append(I)
        R_arr.append(R)
        incidence.append(new_infected)

        mu = rho * new_infected # mean (reporting probability * incidences)
        mu2 = np.maximum(mu, 1e-6)
        p = phi / (phi + mu2) 
        rvs = np.random.negative_binomial(phi, p)
        noisy.append(rvs)

    return {"S": S_arr, "I": I_arr, "R": R_arr, "incidence": incidence, "noisy": noisy}


# Summary statistics of posterior distribution
def summary(param_1, param_2, posterior_1, posterior_2, decimal):
    summary = {'Parameter': [param_1, param_2],
           'Mean': [az.mean(posterior_1, round_to=decimal), az.mean(posterior_2, round_to=decimal)],
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