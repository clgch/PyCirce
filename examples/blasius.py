import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import random
import copy
import jax.numpy as jnp
from kernax.kernels import Energy
from kernax import KernelHerding
import matplotlib.pyplot as plt 
from matplotlib import rc
import seaborn as sns

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 22})
rc('lines', linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"

def blasius(x):
    """
    Helper function that computes the shear coefficient using Blasius correlation
    """

    P = x[:, 0]
    G = x[:, 1]
    T = x[:, 2]
    D = x[:, 3]

    mu_f = cp.CoolProp.PropsSI('V', 'T', T, 'P', P, 'HEOS::Water') 
    
    return np.log10(0.316) - 0.25 * np.log10((G * D) / mu_f)

def generate_data_blasius(data, seed):
    """
    Generate shear coeff data according to the design of experiments with Blasius coefficient 
    """
    np.random.seed(seed)

    n = data.shape[0]

    P = data[:, 0]
    G = data[:, 1]
    T = data[:, 2]
    D = data[:, 3]

    mu_f = cp.CoolProp.PropsSI('V', 'T', T, 'P', P, 'HEOS::Water')

    gamma_true = np.diag(np.array([.05, 0.1]))
    mu_true = np.array([np.log10(0.316), -0.25])

    theta = np.random.multivariate_normal(mu_true, gamma_true, size=n)

    return theta[:, 0] + theta[:, 1] * np.log10((G * D) / mu_f)

def compute_h(x):

    n = x.shape[0]

    P = x[:, 0]
    G = x[:, 1]
    T = x[:, 2]
    D = x[:, 3]

    mu_f = cp.CoolProp.PropsSI('V', 'T', T, 'P', P, 'HEOS::Water') 

    return np.repeat(1.0, n), np.log10((G * D) / mu_f)

np.random.seed(0)

n = 30

T = np.array([25, 110, 250])
P = np.array([60, 90, 110, 130])

data = np.zeros((n, 4))

data[:, 0] = P[np.random.choice(4, size=n, replace=True)] * 100000
data[:, 1] = np.random.uniform(500, 3500, size=n)

data[:, 2] = T[np.random.choice(3, size=n, replace=True)]
data[:, 2] = data[:, 2] + 273.15

data[:, 3] = np.repeat(4e-3, n) 

std = np.zeros((n, 4))

std[:, 0] = np.repeat(50, n)
std[:, 1] = 0.02 * data[:, 1] 
std[:, 2] = 0.02 * data[:, 2] / np.sqrt(3) 
std[:, 3] = 2e-3 / np.sqrt(3)

data[:, 0] = data[:, 0] + np.random.normal(0, std[:, 0], size=n)
data[:, 1] = data[:, 1] + np.random.normal(0, std[:, 1], size=n)
data[:, 2] = data[:, 2] + np.random.uniform(-np.sqrt(3) * std[:, 2] , np.sqrt(3) * std[:, 2], size=n)
data[:, 3] = data[:, 3] + np.random.uniform(-np.sqrt(3) * std[:, 3] , np.sqrt(3) * std[:, 3], size=n)

h = np.zeros((2, n))

h[:, :] = compute_h(data)

print(np.linalg.eigvals(h @ h.T))
print(np.linalg.inv(h @ h.T))

f_blasius = blasius(data)

sig_eps = np.random.normal(0, 0.01, n) * blasius(data)
f_blasius_exp = generate_data_blasius(data, 0) + sig_eps

#plt.plot(f_blasius, f_blasius_exp, '*')
#plt.show()

circe_diag = pyc.CirceEMdiag(initial_mean=[np.log10(0.316), -0.25], initial_cov=np.identity(2), h=h, z_exp=f_blasius_exp, z_nom=np.repeat(0, n), sig_eps=sig_eps, niter=15000, tolerance=1e-4)

circe_ecme_diag = pyc.CirceECMEdiag(initial_mean=[np.log10(0.316), -0.25], initial_cov=np.identity(2), h=h, z_exp=f_blasius_exp, z_nom=np.repeat(0, n), sig_eps=sig_eps, niter=15000, tolerance=1e-4)

mu1, gamma1, loglik, err_cov1, err_mean1 = circe_diag.estimate()
mu2, gamma2, loglik2, err_cov2, err_mean2 = circe_ecme_diag.estimate()

print(f"iterations EM = {len(err_mean1)}")
print(f"err mean (EM CIRCE) = {err_mean1[-1]}")
print(f"err cov (EM CIRCE) = {err_cov1[-1]}")

print(f"iterations ECME = {len(err_mean2)}")
print(f"err mean (ECME CIRCE) = {err_mean2[-1]}")
print(f"err cov (ECME CIRCE) = {err_cov2[-1]}")

print(f"mu (EM CIRCE) = {mu1[-1]}")
print(f"gamma (EM CIRCE) = {np.diag(gamma1[-1])}")

print(f"mu (ECME CIRCE) = {mu2[-1]}")
print(f"gamma (ECME CIRCE) = {np.diag(gamma2[-1])}")