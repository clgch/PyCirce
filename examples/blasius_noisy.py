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

def grad_log_blasius_corr(x):
    step = 1e-10
    nabla = np.zeros((x.shape[0], 4))

    for i in range(3):
        xmin = copy.deepcopy(x)
        xmin[:, i] = xmin[:, i] - step

        ymin = blasius(xmin)

        xmax = copy.deepcopy(x)
        xmax[:, i] = xmax[:, i] + step

        ymax = blasius(xmax)

        nabla[:, i] = (ymax - ymin) / (2 * step)

    return nabla

def generate_data_blasius_noisy(mu, gamma, data, std, seed):
    np.random.seed(seed)

    n = data.shape[0]

    P = data[:, 0] + np.random.normal(0, std[:, 0], size=n)
    G = data[:, 1] + np.random.normal(0, std[:, 1], size=n)
    T = data[:, 2] + np.random.uniform(-np.sqrt(3) * std[:, 2] , np.sqrt(3) * std[:, 2], size=n)
    D = data[:, 3] + np.random.uniform(-np.sqrt(3) * std[:, 3] , np.sqrt(3) * std[:, 3], size=n)

    mu_f = cp.CoolProp.PropsSI('V', 'T', T, 'P', P, 'HEOS::Water') 
    theta = np.random.multivariate_normal(mu, gamma, size=n)

    return theta[:, 0] + theta[:, 1] * np.log10((G * D) / mu_f)

np.random.seed(0)

n = 100

T = np.array([25, 110, 250])
P = np.array([60, 90, 110, 130])

data = np.zeros((n, 4))

data[:, 0] = P[np.random.choice(4, size=n, replace=True)] * 100000
data[:, 1] = np.random.uniform(250, 3000, size=n)

data[:, 2] = np.random.uniform(20, 40, size=n) * (data[:, 0] < 110 * 100000) + np.random.uniform(20, 300, size=n) * (data[:, 0] > 100 * 100000)
data[:, 2] = data[:, 2] + 273.15

data[:, 3] = np.repeat(4e-3, n) 

std = np.zeros((n, 4))

std[:, 0] = np.repeat(30, n)
std[:, 1] = 0.02 * data[:, 1] 
std[:, 2] = 0.02 * data[:, 2] / np.sqrt(3) 
std[:, 3] = 2e-3 / np.sqrt(3)

data[:, 0] = data[:, 0] + np.random.normal(0, std[:, 0], size=n)
data[:, 1] = data[:, 1] + np.random.normal(0, std[:, 1], size=n)
data[:, 2] = data[:, 2] + np.random.uniform(-np.sqrt(3) * std[:, 2] , np.sqrt(3) * std[:, 2], size=n)
data[:, 3] = data[:, 3] + np.random.uniform(-np.sqrt(3) * std[:, 3] , np.sqrt(3) * std[:, 3], size=n)

nabla_design = grad_log_blasius_corr(data) 

std_varf = np.sqrt(np.sum(nabla_design ** 2  * std ** 2, axis=1) + 0.01 ** 2 * blasius(data) ** 2) 

# 2) Monte-Carlo based propagation of uncertainties  

q975 = np.zeros(n)
q0025 = np.zeros(n)
h_med = np.zeros(n)

N = 1000

for i in range(n): 

    m = data[i, :]
    s = std[i, :]

    X_MC = np.zeros((N, 4))

    X_MC[:, 0] = m[0] + np.random.normal(0, s[0], size=N)
    X_MC[:, 1] = m[1] + np.random.normal(0, s[1], size=N)
    X_MC[:, 2] = m[2] + np.random.uniform(-s[2] * np.sqrt(3) , s[2] * np.sqrt(3), size=N)
    X_MC[:, 3] = m[3] + np.random.uniform(-s[3] * np.sqrt(3) , s[3] * np.sqrt(3), size=N)
    
    z_MC =  blasius(X_MC) + np.random.normal(0, 0.01 * np.abs(blasius(X_MC)), size=N)

    q975[i] = np.quantile(z_MC, 0.975)
    q0025[i] = np.quantile(z_MC, 0.025)
    h_med[i] = np.median(z_MC)

f_nom = blasius(data)

plt.figure(figsize=(14, 8)) 
plt.errorbar(f_nom, h_med, yerr=(h_med - q0025, q975 - h_med), fmt='o', capsize=5, color=bleuEDF, alpha=0.7, label=r"Monte-Carlo")
plt.errorbar(f_nom,f_nom, yerr=1.96 * std_varf, fmt='o', capsize=5, color=rougeCEA, alpha=0.7, label=r"Variance formula") 
#plt.plot(Nu_nom, Nu_nom, '+', color=bleuEDF, label=r"$\log_{10}(Nu)^{\rm DB}$")
#plt.plot(Nu_nom, h_med, '*', color=bleuEDF, alpha=0.7)
#plt.plot(np.arange(n), Nu_nom - h_med, '*',  color=bleuEDF, alpha=0.7)
plt.xlabel(r"$\log_{10}(f)$")
plt.ylabel(r"$g_{\rm Blasius}(x)$")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Noisy CIRCE estimation on simulated data 

N = 1000

h_noise = np.zeros((2, N, n))

for i in range(n): 

    m = data[i, :]
    s = std[i, :]

    X_MC = np.zeros((N, 4))

    X_MC[:, 0] = m[0] + np.random.normal(0, s[0], size=N)
    X_MC[:, 1] = m[1] + np.random.normal(0, s[1], size=N)
    X_MC[:, 2] = m[2] + np.random.uniform(-s[2] * np.sqrt(3) , s[2] * np.sqrt(3), size=N)
    X_MC[:, 3] = m[3] + np.random.uniform(-s[3] * np.sqrt(3) , s[3] * np.sqrt(3), size=N)

    h_noise[:, :, i] = compute_h(X_MC)


gamma_true = np.diag(np.array([1, 0.1]))
mu_true = np.array([np.log10(0.316), -0.25])

sig_eps = np.random.normal(0, 0.01 * np.abs(blasius(data)), size=n)
f_blasius_exp = generate_data_blasius_noisy(mu_true, gamma_true, data, std, 0) + sig_eps


noisy_circe = pyc.NoisyCirceDiag(initial_mean=[np.log10(0.316), -0.25], initial_cov=1e-1 * np.identity(2), h=h_noise, z_exp=f_blasius_exp, z_nom=np.repeat(0, n), sig_eps=sig_eps, niter=15000)


h = np.zeros((2, n))

h[:, :] = compute_h(data)
sig_varf = np.sqrt(np.sum(nabla_design ** 2  * std ** 2, axis=1) + 0.01 ** 2 * blasius(data) ** 2)

circe_diag = pyc.CirceEMdiag(initial_mean=[np.log10(0.316), -0.25], initial_cov=1e-1 * np.identity(2), h=h, z_exp=f_blasius_exp, z_nom=np.repeat(0, n), sig_eps=sig_varf, niter=15000)


start = time.time()
mu1, gamma1, err_cov1, err_mean1 = noisy_circe.estimate()
stop = time.time()

print(f"Compution MC EM = {stop - start} sec")

start = time.time()
mu2, gamma2, loglik, err_cov2, err_mean2 = circe_diag.estimate()
stop = time.time()
print(f"Compution EM = {stop - start} sec")

print(f"iterations MC EM = {len(err_mean1)}")
print(f"err mean (MC EM) = {err_mean1[-1]}")
print(f"err cov (MC EM) = {err_cov1[-1]}")

print(f"iterations EM = {len(err_mean2)}")
print(f"err mean (EM) = {err_mean2[-1]}")
print(f"err cov (EM) = {err_cov2[-1]}")

print(f"mu (MC EM) = {mu1[-1]}")
print(f"gamma (MC EM) = {np.diag(gamma1[-1])}")

print(f"mu (EM) = {mu2[-1]}")
print(f"gamma (EM) = {np.diag(gamma2[-1])}")