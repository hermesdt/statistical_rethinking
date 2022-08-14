import random
import numpy as np
from scipy.stats.distributions import bernoulli
import pandas as pd
import pymc as pm

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def sim_happiness(seed=1977 , N_years=1000 , max_age=65 , N_births=20 , aom=18):
    random.seed(seed)
    H, M, A = np.array([]), np.array([]), np.array([])
    for t in range(N_years):
        A += 1
        A = np.concatenate((A, np.ones(N_births)))
        H = np.concatenate((H, np.linspace(-2, 2, num=N_births)))
        M = np.concatenate((M, np.zeros(N_births)))

        for i in range(len(A)):
            if A[i] >= aom and M[i] == 0:
                M[i] = bernoulli.rvs(inv_logit(H[i]-4), size=1)
        
        deaths = A > max_age
        if len(deaths) > 0:
            A = A[~deaths]
            H = H[~deaths]
            M = M[~deaths]
                
    return pd.DataFrame({"age": A, "happiness": H, "married": M})

def link(model, trace, progressbar=False, var_names=None, **datas):
    with model:
        for data, value in datas.items():
            model.set_data(data, value)
        return pm.sample_posterior_predictive(trace, progressbar=progressbar, var_names=var_names)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
