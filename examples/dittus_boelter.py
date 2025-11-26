import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import random
import copy
from scipy.stats import invwishart
import matplotlib.pyplot as plt 
from matplotlib import rc

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 22})
rc('lines', linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"


## This code generates virtual samples from the Dittus Bolter correlation for the Nusselt number: Nu = 0.0023 * Re^0.8 * Pr^0.3, Experimental values of Re and Pr are taken from Becker post CHF experimental campaign.
#
# The intercept C = log(0.0023) is supposed fixed, the statistical model is 
# 
#log(Nu) - C = theta_1 * log(Re) + theta_2 * log(Pr) + epsilon 
#
# epsilon is a standard Gaussian of std 0.1
# 
# theta = [theta_1, theta_1] is a Gaussian centered in [0.8, 0.3] with a covariance matrix # # # sampled from an inverse Wishart distribution whose hyperparameters are fitted on a dataset of # size 5.


def dittus_boelter_corr(x):
    """
    Helper function that computes the Nusselt number using the Dittus Boelter correlation
    """

    P = x[:, 0]
    G = x[:, 1]
    Tf = x[:, 2]

    lambda_f = cp.CoolProp.PropsSI('L', 'T', Tf, 'P', P, 'IF97::Water') * 1e-3
    mu_f = cp.CoolProp.PropsSI('V', 'T', Tf, 'P', P, 'IF97::Water') 
    c_p = cp.CoolProp.PropsSI('C', 'T', Tf, 'P', P, 'IF97::Water')

    return 0.023 * (G * 8 * 1e-3 / mu_f) ** 0.8 * (c_p * mu_f / lambda_f) ** 0.4

def grad_log_db_corr(x):
    step = 1e-5
    nabla = np.zeros((x.shape[0],3))

    for i in range(3):
        xmin = copy.deepcopy(x)
        xmin[:, i] = xmin[:, i] - step

        ymin = np.log10(dittus_boelter_corr(xmin))

        xmax = copy.deepcopy(x)
        xmax[:, i] = xmax[:, i] + step

        ymax = np.log10(dittus_boelter_corr(xmax))

        nabla[:, i] = (ymax - ymin) / (2 * step)

    return nabla


np.random.seed(0)
random.seed()

# 1) Variance formula for uncertainties propagation on the  

X = pd.read_csv("./examples/becker_design.csv").values[:, 1:]

n = X.shape[0]

std_X = np.zeros((n, 3))

std_X[:, 0] = np.repeat(20, n)
std_X[:, 1] = 0.02 * X[:, 1]
std_X[:, 2] = 0.02 * X[:, 2]


nabla_design = grad_log_db_corr(X)

std_log_nu = np.sqrt(np.sum(nabla_design ** 2  * std_X ** 2, axis=1) + np.log10(dittus_boelter_corr(X)) ** 2 * 0.01 ** 2) 

print(std_log_nu.shape)

Nu_exp = pd.read_csv("./examples/becker_experiments.csv").values[:, 1]
Nu_nom = np.log10(dittus_boelter_corr(X))

plt.figure(figsize=(14, 8))
plt.errorbar(Nu_nom, Nu_exp, yerr=2 * std_log_nu, fmt='o', capsize=5, color=rougeCEA, label=r"$\log_{10}(Nu)^{\rm exp}$")
plt.plot(Nu_nom, Nu_nom, '+', color=bleuEDF, label=r"$\log_{10}(Nu)^{\rm DB}$")
plt.xlabel(r"$\log_{10}(Nu)^{\rm DB}$")
plt.ylabel(r"$\log_{10}(Nu)^{\rm exp}$")
plt.legend()
plt.tight_layout()
plt.show()

h = np.zeros((2, n))

#h[0, :] = np.repeat(1, n)
h = pd.read_csv("./examples/becker_experiments.csv").values[:, 2:].T

em_diag = pyc.CirceEMdiag(initial_mean=[0.8, 0.4], initial_cov=np.identity(2), h=h, z_exp=Nu_exp, z_nom=Nu_nom, sig_eps=std_log_nu, niter=3000)

mean, cov, loglik = em_diag.estimate()

print(f"mu = {mean[-1]}")
print(f"sig2 = {np.diag(cov[-1])}")


s = list(range(315))
random.shuffle(s)
idx = s[:40]

# Load 1 values of log(Re) and log(Pr) 
X = pd.read_csv("./examples/becker_experiments.csv").values[idx, 2:]

cov = np.linalg.inv((X.T @ X)) * 0.1 ** 2

std_devs = np.sqrt(np.diag(cov))
outer_product = np.outer(std_devs, std_devs)

corr = cov / outer_product

nu = 5 + 2 + 2  
Psi = cov 

# Sample a covariance matrix for theta
#gamma_true = invwishart.rvs(nu, Psi)
gamma_true = np.diag(np.array([2,  7]))
mu_true = np.array([0.8, 0.3])

print(gamma_true)

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


# ecme = pyc.CirceECME(initial_mean=[0.8, 0.3], initial_cov=np.identity(2), h=h, z_exp=Nu_exp, z_nom=Nu_nom, sig_eps=np.repeat(0.1, 15))

# _, _, loglik = ecme.estimate()

# plt.plot(range(1001), loglik)
# plt.show()