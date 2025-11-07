import numpy as np
import pycirce as pyc
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 22})
rc('lines', linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"

def f(seed):

    np.random.seed(seed)

    initial_mean = np.random.multivariate_normal(np.array([1.05 * 55, 0.95 * 30]), 1 * np.eye(2), size=1)

    initial_cov = np.array([[1, 0], [0, 1]])

    ecme = pyc.CirceECME(initial_mean=initial_mean, initial_cov=initial_cov, h=X, z_exp=z_exp, z_nom=z_nom, sig_eps=np.repeat(0.1, n), niter=niter)

    mean, cov, loglik = ecme.estimate()

    return mean[-1], cov[-1], loglik[-1]

# Simple toy case: Y_i = X_i^T lambda_i + e_i;  X ~ N(0, I_p), lambda_i \sim N(mu, Gamma), mu = [1, 2], e_i ~ N(0, 1).abs

np.random.seed(10)

p = 2
n = 15
niter = 1000

mu_true = np.array([55, 30])

rho = 0.2
v1 = (0.1 * 55) ** 2 
v2 = (0.1 * 30) ** 2
gamma_true = np.array([[v1, rho * np.sqrt(v1 * v2)], [rho * np.sqrt(v1 * v2), v2]])

lambda_sample = np.random.multivariate_normal(mu_true, gamma_true, size=n)
X = np.random.normal(size=(p, n)) 
epsilon = np.random.normal(loc=0, scale=0.1, size=n)

#z_exp = np.sum(X.T * lambda_sample, axis=1) + epsilon
#z_nom = np.sum(X.T * np.array([1, 2]), axis=1)

z_exp = np.zeros(n)
z_nom = np.zeros(n)
for i in range(n):
    z_exp[i] = np.sum(X.T[i] * lambda_sample[i]) + epsilon[i] 
    z_nom[i] = 0

print(f"cov0 = {gamma_true}")
print(f"mean0 = {mu_true}")

nrep = 100
seed = range(nrep)  # Example arguments

with ProcessPoolExecutor(max_workers=10) as executor:
    res = list(executor.map(f, seed))

means = []
covs = []
ll = []

for i in range(nrep):
    means += [res[i][0].flatten()]
    covs += [res[i][1].flatten()]
    ll += [res[i][2][0]]

means = np.array(means)
covs = np.array(covs)

plt.hist(ll)
plt.show()

print(f"median estimate m1 = {np.median(means[:, 0])}")
print(f"std dev m1 = {np.std(means[:, 0])}\n")

print(f"median estimate m2 = {np.median(means[:, 1])}")
print(f"std dev m2 = {np.std(means[:, 1])}\n")

print(f"true v1 = {v1}")
print(f"median estimate v1 = {np.median(covs[:, 0])}")
print(f"std dev v1 = {np.std(covs[:, 0])}\n")

print(f"true v2 = {v2}")
print(f"median estimate v2 = {np.median(covs[:, 3])}")
print(f"std dev sig2 = {np.std(covs[:, 3])}\n")

print(f"true cov12 = {rho * np.sqrt(v1 * v2)}")
print(f"median estimate cov12 = {np.median(covs[:, 1])}")
print(f"std dev cov12 = {np.std(covs[:, 1])}\n")
    

    # it = len(np.array(mean)[:, 0, :])

    # plt.plot(range(it), np.array(mean)[:, 0, :], label=r"$\mu_1$")
    # plt.plot(range(it), np.array(mean)[:, 1, :], label=r"$\mu_2$")
    # plt.legend()
    # plt.show()

    # plt.plot(range(it-1), np.array(loglik)[:, 0])
    # plt.show()