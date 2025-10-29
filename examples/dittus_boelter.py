import numpy as np
import pandas as pd
import pycirce as pyc
import random
from scipy.stats import invwishart


## This code generates virtual samples from the Dittus Bolter correlation for the Nusselt number: Nu = 0.0023 * Re^0.8 * Pr^0.3, Experimental values of Re and Pr are taken from Becker post CHF experimental campaign.
#
# The intercept C = log(0.0023) is supposed fixed, the statistical model is 
# 
#log(Nu) - C = theta_1 * log(Re) + theta_2 * log(Pr) + epsilon 
#
# epsilon is a standard Gaussian of std 0.1
# 
# theta = [theta_1, theta_1] is a Gaussian centered in [0.8, 0.3] with a covariance matrix # # # sampled from an inverse Wishart distribution whose hyperparameters are fitted on a dataset of # size 5.


np.random.seed(0)

s = list(range(315))
random.shuffle(s)
idx = s[:5]

# Load 1 values of log(Re) and log(Pr) 
X = pd.read_csv("./examples/becker_experiments.csv").values[idx, 2:]

cov = np.linalg.inv((X.T @ X)) * 0.1 ** 2

std_devs = np.sqrt(np.diag(cov))
outer_product = np.outer(std_devs, std_devs)

corr = cov / outer_product

nu = 5 + 2 + 2  
Psi = cov 

# Sample a covariance matrix for theta
gamma_true = invwishart.rvs(nu, Psi)
mu_true = np.array([0.8, 0.3])

# Load 15 values of log(Re) and log(Pr) 
s = list(range(315))
random.shuffle(s)
idx = s[:15]

X = pd.read_csv("./examples/becker_experiments.csv").values[idx, 2:]

theta_sample = np.random.multivariate_normal(mu_true, gamma_true, size=15)
epsilon = np.random.normal(loc=0, scale=0.1, size=15)

Nu_exp = np.sum(X * theta_sample, axis=1) + np.log10(0.0023) + epsilon
Nu_nom = np.sum(X * np.array([0.8, 0.3]), axis=1) + np.log10(0.0023)

h = X.T

ecme = pyc.CirceECME(initial_mean=[0.8, 0.3], initial_cov=np.identity(2), h=h, z_exp=Nu_exp, z_nom=Nu_nom, sig_eps=np.repeat(0.1, 15))

ecme.estimate()