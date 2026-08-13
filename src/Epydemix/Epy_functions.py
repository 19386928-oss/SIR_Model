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