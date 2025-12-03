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
import time

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

def generate_data_blasius(mu, gamma, data, seed):
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

    theta = np.random.multivariate_normal(mu, gamma, size=n)

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

n = 100

T = np.array([25, 110, 250])
P = np.array([60, 90, 110, 130])

data = np.zeros((n, 4))

data[:, 0] = P[np.random.choice(4, size=n, replace=True)] * 100000
data[:, 1] = np.random.uniform(250, 3000, size=n)

#data[:, 2] = T[np.random.choice(3, size=n, replace=True)]
data[:, 2] = np.random.uniform(20, 40, size=n) * (data[:, 0] < 110 * 100000) + np.random.uniform(20, 300, size=n) * (data[:, 0] > 100 * 100000)
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

# sns.pairplot(pd.DataFrame(data[:, :3], columns=["P", "G", "T"]))
# plt.show()

h = np.zeros((2, n))

h[:, :] = compute_h(data)

# plt.hist(h[1, :], density=True)
# plt.show()

# print(np.linalg.eigvals(h @ h.T))
# print(np.linalg.inv(h @ h.T))

# f_blasius = blasius(data)

# sig_eps = np.random.normal(0, 0.01, n) * blasius(data)
# f_blasius_exp = generate_data_blasius(data, 0) + sig_eps

#plt.plot(f_blasius, f_blasius_exp, '*')
#plt.show()

n_rep = 200

mu_em = np.zeros((n_rep+1, 2))
gamma_em = np.zeros((n_rep+1, 2))

mu_ecme = np.zeros((n_rep+1, 2))
gamma_ecme = np.zeros((n_rep+1, 2))

gamma_true = np.diag(np.array([.1, .1]))
mu_true = np.array([np.log10(0.316), -0.25])

mu_em[0, :] = mu_true
gamma_em[0, :] = np.diag(gamma_true)

mu_ecme[0, :] = mu_true
gamma_ecme[0, :] = np.diag(gamma_true)

for it in range(1, n_rep+1):
    print(f"it = {it}")
    sig_eps = np.random.normal(0, 0.01, n) * blasius(data)
    f_blasius_exp = generate_data_blasius(mu_true, gamma_true, data, it) + sig_eps

    circe_diag = pyc.CirceEMdiag(initial_mean=[np.log10(0.316), -0.25], initial_cov=1e-1 * np.identity(2), h=h, z_exp=f_blasius_exp, z_nom=np.repeat(0, n), sig_eps=sig_eps, niter=15000, tolerance=1e-4)
    circe_diag_ecme = pyc.CirceECMEdiag(initial_mean=[np.log10(0.316), -0.25], initial_cov=1e-1 * np.identity(2), h=h, z_exp=f_blasius_exp, z_nom=np.repeat(0, n), sig_eps=sig_eps, niter=15000, tolerance=1e-4)

    mu1, gamma1, loglik, err_cov1, err_mean1 = circe_diag_ecme.estimate()
    mu2, gamma2, loglik2, err_cov2, err_mean2 = circe_diag.estimate()

    mu_em[it, :] = np.array(mu2[-1])[:, 0]
    gamma_em[it, :] = np.diag(gamma2[-1])

    mu_ecme[it, :] = np.array(mu1[-1])[:, 0]
    gamma_ecme[it, :] = np.diag(gamma1[-1])

pd.DataFrame(mu_em).to_csv(f"./results/mu_em_nrep_{n_rep}.csv", index=False)
pd.DataFrame(gamma_em).to_csv(f"./results/gamma_em_nrep_{n_rep}.csv", index=False)
pd.DataFrame(mu_ecme).to_csv(f"./results/mu_ecme_nrep_{n_rep}.csv", index=False)
pd.DataFrame(gamma_ecme).to_csv(f"./results/gamma_ecme_nrep_{n_rep}.csv", index=False)

# start2 = time.time()
# mu2, gamma2, loglik2, err_cov2, err_mean2 = circe_diag.estimate()
# stop2 = time.time()
# print(f"time spent = {stop2 - start2}")


# start1 = time.time()
# mu1, gamma1, loglik, err_cov1, err_mean1 = circe_diag_ecme.estimate()
# stop1 = time.time()
# print(f"time spent = {stop1 - start1}")


# print(f"iterations EM = {len(err_mean1)}")
# print(f"err mean (EM CIRCE) = {err_mean1[-1]}")
# print(f"err cov (EM CIRCE) = {err_cov1[-1]}")

# print(f"iterations ECME = {len(err_mean2)}")
# print(f"err mean (ECME CIRCE) = {err_mean2[-1]}")
# print(f"err cov (ECME CIRCE) = {err_cov2[-1]}")

# print(f"mu (EM CIRCE) = {mu1[-1]}")
# print(f"gamma (EM CIRCE) = {np.diag(gamma1[-1])}")

# print(f"mu (ECME CIRCE) = {mu2[-1]}")
# print(f"gamma (ECME CIRCE) = {np.diag(gamma2[-1])}")