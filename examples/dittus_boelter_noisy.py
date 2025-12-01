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


## This code generates virtual samples from the Dittus Bolter correlation for the Nusselt number: Nu = 0.0023 * Re^0.8 * Pr^0.3, Experimental values of Re and Pr are taken from a sub sample of Becker post CHF experimental campaign.
#
# The intercept C = log(0.0023) is supposed fixed, the statistical model is 
# 
#log(h) = C +  theta_1 * log(D/lambda)  + theta_2 * log(Re) + theta_3 * log(Pr) + epsilon 
#
# epsilon is a standard Gaussian of std 0.1

def dittus_boelter_corr(x):
    """
    Helper function that computes the Nusselt number using the Dittus Boelter correlation
    """

    P = x[:, 0]
    G = x[:, 1]
    Tp = x[:, 2]
    D = x[:, 3]

    Tsat = cp.CoolProp.PropsSI('T', 'P', P, 'Q', 0, 'HEOS::Water') 
    Tf = (Tsat + Tp) / 2

    lambda_f = cp.CoolProp.PropsSI('L', 'T', Tf, 'P', P, 'HEOS::Water') * 1e-3
    mu_f = cp.CoolProp.PropsSI('V', 'T', Tf, 'P', P, 'HEOS::Water') 
    c_p = cp.CoolProp.PropsSI('C', 'T', Tf, 'P', P, 'HEOS::Water')

    return 0.023 * (lambda_f/ D)  * (G * D / mu_f) ** 0.8 * (c_p * mu_f / lambda_f) ** 0.4

def grad_log_db_corr(x):
    step = 1e-10
    nabla = np.zeros((x.shape[0], 4))

    for i in range(3):
        xmin = copy.deepcopy(x)
        xmin[:, i] = xmin[:, i] - step

        ymin = np.log10(dittus_boelter_corr(xmin))

        xmax = copy.deepcopy(x)
        xmax[:, i] = xmax[:, i] + step

        ymax = np.log10(dittus_boelter_corr(xmax))

        nabla[:, i] = (ymax - ymin) / (2 * step)

    return nabla

def generate_data_becker(data, std, seed):
    """
    Generate postCHF data according to Becker's experimental dataset 
    """
    np.random.seed(seed)

    n = data.shape[0]

    P = data[:, 0] + np.random.normal(0, std[:, 0], size=n)
    G = data[:, 1] + np.random.normal(0, std[:, 1], size=n)
    Tp = data[:, 2] + np.random.uniform(-std[:, 2] * np.sqrt(3) , std[:, 2] * np.sqrt(3), size=n)
    D = data[:, 3] + np.random.uniform(-std[:, 3] * np.sqrt(3) , std[:, 3] * np.sqrt(3), size=n)

    Tsat = cp.CoolProp.PropsSI('T', 'P', P, 'Q', 0, 'HEOS::Water') 
    Tf = (Tsat + Tp) / 2

    lambda_f = cp.CoolProp.PropsSI('L', 'T', Tf, 'P', P, 'HEOS::Water') * 1e-3
    mu_f = cp.CoolProp.PropsSI('V', 'T', Tf, 'P', P, 'HEOS::Water') 
    c_p = cp.CoolProp.PropsSI('C', 'T', Tf, 'P', P, 'HEOS::Water')

    gamma_true = np.diag(np.array([0.1, 0.01, 0.01]))
    mu_true = np.array([1, 0.8, 0.4])

    theta = np.random.multivariate_normal(mu_true, gamma_true, size=n)

    return 0.023 * np.power((lambda_f / D), theta[:, 0])  * np.power((G * D / mu_f), theta[:, 1]) * np.power((c_p * mu_f / lambda_f), theta[:, 2])

def compute_h(x):

    P = x[:, 0]
    G = x[:, 1]
    Tp = x[:, 2]
    D = x[:, 3]

    Tsat = cp.CoolProp.PropsSI('T', 'P', P, 'Q', 0, 'HEOS::Water') 
    Tf = (Tsat + Tp) / 2

    lambda_f = cp.CoolProp.PropsSI('L', 'T', Tf, 'P', P, 'HEOS::Water') * 1e-3
    mu_f = cp.CoolProp.PropsSI('V', 'T', Tf, 'P', P, 'HEOS::Water') 
    c_p = cp.CoolProp.PropsSI('C', 'T', Tf, 'P', P, 'HEOS::Water')

    return np.log10(lambda_f / D), np.log10((G * D / mu_f)), np.log10((c_p * mu_f / lambda_f))


np.random.seed(0)
random.seed()

# Perform kernel herding (support points) to select 20 inputs variables from the 315-sized Becker's dataset

n = 20

X = np.zeros((315, 4))
X[:, :3] = pd.read_csv("./examples/becker_design.csv").values[:, 1:]
X[:, 3] = np.repeat(8e-3, 315)

jax_X = jnp.array(X)

quantization_fn = KernelHerding(jax_X, kernel_fn=Energy)
idx = quantization_fn(m = n)

X_sp = X[idx, :]
support_points = pd.DataFrame(X[idx, :])

std_X = np.zeros((n, 4))

std_X[:, 0] = np.repeat(50, n)
std_X[:, 1] = 0.02 * X_sp[:, 1] 
std_X[:, 2] = 0.1 * X_sp[:, 2] / np.sqrt(3) 
std_X[:, 3] = 4e-3 / np.sqrt(3)

sig_eps = np.random.normal(0, 0.01, n) * np.log10(dittus_boelter_corr(X_sp))
z_sim = np.log10(generate_data_becker(X_sp, std_X, 0)) + sig_eps
h = np.log10(dittus_boelter_corr(X_sp))


# plt.plot(h, z_sim, '*')
# plt.show()

#sns.pairplot(support_points)
#plt.show()



# 1) Variance formula for uncertainties propagation on the  


std_X = np.zeros((n, 4))

std_X[:, 0] = np.repeat(20, n)
std_X[:, 1] = 0.002 * X_sp[:, 1] 
std_X[:, 2] = 0.002 * X_sp[:, 2] / np.sqrt(3) 
std_X[:, 3] = 4e-3 / np.sqrt(3)

#std_X = 1e-2 * std_X

nabla_design = grad_log_db_corr(X_sp)

# + np.log10(dittus_boelter_corr(X)) ** 2 * 0.01 ** 2

std_log_nu = np.sqrt(np.sum(nabla_design ** 2  * std_X ** 2, axis=1)) 
#+ np.log10(dittus_boelter_corr(X_sp)) ** 2 * 0.01 ** 2) 

# 2) Monte-Carlo based propagation of uncertainties  

q975 = np.zeros(n)
q0025 = np.zeros(n)
h_med = np.zeros(n)

N = 1000

for i in range(n): 

    m = X_sp[i, :]
    s = std_X[i, :]


    X_MC = np.zeros((N, 4))

    X_MC[:, 0] = m[0] + np.random.normal(0, s[0], size=N)
    X_MC[:, 1] = m[1] + np.random.normal(0, s[1], size=N)
    X_MC[:, 2] = m[2] + np.random.uniform(-s[2] * np.sqrt(3) , s[2] * np.sqrt(3), size=N)
    X_MC[:, 3] = 8e-3 + np.random.uniform(-s[3] * np.sqrt(3) , s[3] * np.sqrt(3), size=N)
    
    z_MC =  np.log10(dittus_boelter_corr(X_MC)) 
    #+ np.random.normal(0, 0.01 * np.log10(dittus_boelter_corr(X_MC)), size=N)

    q975[i] = np.quantile(z_MC, 0.975)
    q0025[i] = np.quantile(z_MC, 0.025)
    h_med[i] = np.median(z_MC)


Nu_exp = pd.read_csv("./examples/becker_experiments.csv").values[:, 1]
Nu_nom = np.log10(dittus_boelter_corr(X_sp)) 

# plt.figure(figsize=(14, 8)) 
# plt.errorbar(Nu_nom, h_med, yerr=(h_med - q0025, q975 - h_med), fmt='o', capsize=5, color=bleuEDF, alpha=0.7, label=r"Monte-Carlo")
# plt.errorbar(Nu_nom, Nu_nom, yerr=1.96 * std_log_nu, fmt='o', capsize=5, color=rougeCEA, alpha=0.7, label=r"Variance formula") 
# #plt.plot(Nu_nom, Nu_nom, '+', color=bleuEDF, label=r"$\log_{10}(Nu)^{\rm DB}$")
# #plt.plot(Nu_nom, h_med, '*', color=bleuEDF, alpha=0.7)
# #plt.plot(np.arange(n), Nu_nom - h_med, '*',  color=bleuEDF, alpha=0.7)
# plt.xlabel(r"$\log_{10}(h)$")
# plt.ylabel(r"$g_{\rm DB}(x)$")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(14, 8)) 
# plt.plot(Nu_nom, (q975 - q0025) / (1.96 * std_log_nu), '*')
# plt.tight_layout()
# plt.show()

 # 3) Noisy CIRCE estimation on simulated data 

N = 500

h_noise = np.zeros((3, N, n))

for i in range(n): 

    m = X_sp[i, :]
    s = std_X[i, :]

    X_MC = np.zeros((N, 4))

    X_MC[:, 0] = m[0] + np.random.normal(0, s[0], size=N)
    X_MC[:, 1] = m[1] + np.random.normal(0, s[1], size=N)
    X_MC[:, 2] = m[2] + np.random.uniform(-s[2] * np.sqrt(3) , s[2] * np.sqrt(3), size=N)
    X_MC[:, 3] = 8e-3 + np.random.uniform(-s[3] * np.sqrt(3) , s[3] * np.sqrt(3), size=N)

    h_noise[:, :, i] = compute_h(X_MC)

sig_eps = np.random.normal(0, 0.01, n) * np.log10(dittus_boelter_corr(X_sp))
z_sim = np.log10(generate_data_becker(X_sp, std_X, 0)) + sig_eps


noisy_circe = pyc.NoisyCirceDiag(initial_mean=[1, 1, 1], initial_cov=np.identity(3), h=h_noise, z_exp=z_sim, z_nom=np.repeat(np.log10(0.023), n), sig_eps=sig_eps, niter=6000)

h = np.zeros((3, n))

h[:, :] = compute_h(X_sp)
sig_varf = np.sqrt(np.sum(nabla_design ** 2  * std_X ** 2, axis=1)+ sig_eps ** 2)

circe_diag = pyc.CirceEMdiag(initial_mean=[1, 1, 1], initial_cov=np.identity(3), h=h, z_exp=z_sim, z_nom=np.repeat(np.log10(0.023), n), sig_eps=sig_varf, niter=6000)

mu1, gamma1, err_cov1, err_mean1 = noisy_circe.estimate()

mu2, gamma2, loglik, err_cov2, err_mean2 = circe_diag.estimate()

print(f"err mean noisy CIRCE = {err_mean1[-1]}")
print(f"err mean CIRCE EM = {err_mean2[-1]}")

print(f"err cov noisy CIRCE = {err_cov1[-1]}")
print(f"err cov CIRCE EM = {err_cov2[-1]}")

print(f"mu (noisy CIRCE) = {mu1[-1]}")
print(f"gamma (noisy CIRCE) = {np.diag(gamma1[-1])}")

print(f"mu (CIRCE EM) = {mu2[-1]}")
print(f"gamma (CIRCE EM) = {np.diag(gamma2[-1])}")