import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import scipy.stats as st
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib import rc
import sys

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"


def toy_case(x):
    """
    Helper function that computes the toy case
    """

    return np.log10(0.316) - 0.25 * x


def generate_data_toycase(mu, gamma, data, seed):
    """
    Generate toy case data according to the design of experiments 
    """
    np.random.seed(seed)

    n = data.shape[0]

    theta = np.random.multivariate_normal(mu, gamma, size=n)

    return theta[:, 0] + theta[:, 1] * data[:, 0] + theta[:, 2] * data[:, 1] 



# T = np.array([25, 110, 250])
# P = np.array([60, 90, 110, 130])

# data = np.zeros((n, 4))

# data[:, 0] = P[np.random.choice(4, size=n, replace=True)] * 100000
# data[:, 1] = np.random.uniform(250, 3000, size=n)

# # data[:, 2] = T[np.random.choice(3, size=n, replace=True)]
# data[:, 2] = np.random.uniform(20, 40, size=n) * (
#     data[:, 0] < 110 * 100000
# ) + np.random.uniform(20, 300, size=n) * (data[:, 0] > 100 * 100000)
# data[:, 2] = data[:, 2] + 273.15

# data[:, 3] = np.repeat(4e-3, n)


# std = np.zeros((n, 4))

# std[:, 0] = np.repeat(50, n)
# std[:, 1] = 0.02 * data[:, 1]
# std[:, 2] = 0.02 * data[:, 2] / np.sqrt(3)
# std[:, 3] = 2e-3 / np.sqrt(3)

# data[:, 0] = data[:, 0] + np.random.normal(0, std[:, 0], size=n)
# data[:, 1] = data[:, 1] + np.random.normal(0, std[:, 1], size=n)
# data[:, 2] = data[:, 2] + np.random.uniform(
#     -np.sqrt(3) * std[:, 2], np.sqrt(3) * std[:, 2], size=n
# )
# data[:, 3] = data[:, 3] + np.random.uniform(
#     -np.sqrt(3) * std[:, 3], np.sqrt(3) * std[:, 3], size=n


# sns.pairplot(pd.DataFrame(data[:, :3], columns=["P", "G", "T"]))
# plt.show()

# print(np.mean(blasius(data)))
# print(np.std(blasius(data)))

# h = np.zeros((2, n))

# h[:, :] = compute_h(data)
# #print(np.linalg.eigvals(h @ h.T))

# print(np.amax(4 ** 2/(h[0, :] ** 2 + h[1, :] ** 2)))

# bmin = 1/3.0 * np.amax(4 ** 2/(h[0, :] ** 2 + h[1, :] ** 2))

# plt.hist(h[1, :], density=True)
# plt.show()

# print(np.linalg.eigvals(h @ h.T))
# print(np.linalg.inv(h @ h.T))

# f_blasius = blasius(data)

# sig_eps = np.random.normal(0, 0.01, n) * blasius(data)
# f_blasius_exp = generate_data_blasius(data, 0) + sig_eps

# plt.plot(f_blasius, f_blasius_exp, '*')
# plt.show()

n_rep = 100

mu_em = np.zeros((n_rep + 1, 3))
gamma_em = np.zeros((n_rep + 1, 3))

mu_ecme = np.zeros((n_rep + 1, 3))
gamma_ecme = np.zeros((n_rep + 1, 3))

mu_reml = np.zeros((n_rep + 1, 3))
gamma_reml = np.zeros((n_rep + 1, 3))

mu_profile = np.zeros((n_rep + 1, 3))
gamma_profile = np.zeros((n_rep + 1, 3))

mu_reml_ridge = np.zeros((n_rep + 1, 3))
gamma_reml_ridge = np.zeros((n_rep + 1, 3))

## BASELINE: h=(1,x1,x2), x~N(0,I2), mu = 0, Gamma=diag(0.5, 0.7, 0.3), sigma2=1

# idx = int(sys.argv[1]) - 1 

# n_array = [5, 10, 15, 20, 25, 30, 500]
# n = n_array[idx]

mu_true = np.array([0, 0, 0])
gamma_true = np.diag(np.array([0.5, 0.7, 0.3]))

n = 20
data = np.random.normal(size=(n, 2))

#exponents = np.random.uniform(-3, 2, size=n)
#data[:, 3] = np.power(10, exponents)


h = np.zeros((3, n))
h[0, :] = np.repeat(1.0, n)
h[1, :] = data[:, 0] 
h[2, :] = data[:, 1]

mean_snr = np.sum(np.diag(gamma_true)[:, None] * h**2) / n

print(f"mean SNR = {mean_snr}") # mean SNR = 1.6
print(f"mean R2 = {mean_snr / (1 + mean_snr)}") # mean R2 = 0.6

mu_em[0, :] = mu_true
gamma_em[0, :] = np.diag(gamma_true)

mu_ecme[0, :] = mu_true
gamma_ecme[0, :] = np.diag(gamma_true)

mu_reml[0, :] = mu_true
gamma_reml[0, :] = np.diag(gamma_true)

mu_profile[0, :] = mu_true
gamma_profile[0, :] = np.diag(gamma_true)

mu_reml_ridge[0, :] = mu_true
gamma_reml_ridge[0, :] = np.diag(gamma_true)


def run_replication(it):
    """
    Run one Monte-Carlo replication: generate noisy data for replication `it`,
    fit EM/ECME/REML/profile estimators, and compute their coverage/interquantile
    diagnostics. Reads module-level globals (n, data, mu_true, gamma_true, cov_rate)
    that are reconstructed identically in each worker process.
    """
    print(f"it = {it}")

    np.random.seed(it)

    sig_eps = np.random.normal(0, 1, n)

    f_toycase_exp = generate_data_toycase(mu_true, gamma_true, data, it) + sig_eps 

    H = h @ h.T
    _, s, _ = np.linalg.svd(H)
    L = np.amax(s)

    # denom = sig_eps**2 + np.sum(np.diag(gamma_true)[:, None] * h**2, axis=0)
    # w = 1.0 / denom
    # H_tilde = (h * w[None, :]) @ h.T
            
    # U, s, Vt = np.linalg.svd(H_tilde)
    # s_pseudo = np.zeros_like(s)
    # non_zero = s > 1e-10
    # s_pseudo[non_zero] = 1.0 / s[non_zero]
    # H_plus = Vt.T @ np.diag(s_pseudo) @ U.T

    # print(H_plus)

    #print(np.mean(h[1, :]) ** 2 / np.var(h[1, :]))

    bmin = 1/3.0 * np.amax((1) ** 2/(h[0, :] ** 2 + h[1, :] ** 2 + h[2, :] ** 2))


    circe_diag = pyc.CirceEMdiag(
        initial_mean=[0.3, -0.3, 0.3],
        initial_cov= bmin * np.identity(3),
        h=h,
        z_exp=f_toycase_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000,
        tolerance=1e-5,
    )

    circe_diag_ecme = pyc.CirceECMEdiag(
        initial_mean=[0.3, -0.3, 0.3],
        initial_cov= bmin * np.identity(3),
        h=h,
        z_exp=f_toycase_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000,
        tolerance=1e-5,
    )

    circe_reml = pyc.CirceREML(
        initial_mean=[0.3, -0.3, 0.3],
        initial_cov= bmin * np.identity(3),
        h=h,
        z_exp=f_toycase_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000
    )

    mu3, gamma3, _ = circe_reml.estimate(n_starts=10)
    alpha = 0.25
    lmbd_opt = (L * np.amin(gamma3))/(4 * (1.0 / alpha - 1) ** 3)

    circe_reml_ridge = pyc.CirceREMLRidge(
            initial_mean=[0.3, -0.3, 0.3],
            initial_cov= bmin * np.identity(3),
            h=h,
            z_exp=f_toycase_exp,
            z_nom=np.repeat(0, n),
            sig_eps=sig_eps,
            lmbd=lmbd_opt,
            gamma0_ridge=gamma3,
            niter=15000
        ) 


    circe_profile = pyc.CirceProfile(
        initial_mean=[0.3, -0.3, 0.3],
        initial_cov= bmin * np.identity(3),
        h=h,
        z_exp=f_toycase_exp ,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000
    )


    mu1, gamma1, _, _ = circe_diag_ecme.estimate(n_starts=10)
    #print(f"iterations ECME = {len(err_mean1)}/15000")

    mu2, gamma2, _, _ = circe_diag.estimate(n_starts=10)
    #print(f"iterations EM = {len(err_mean2)}/15000")

    mu3, gamma3, _ = circe_reml.estimate(n_starts=10)
    mu4, gamma4, _ = circe_profile.estimate(n_starts=10)
    mu5, gamma5, _ = circe_reml_ridge.estimate(n_starts=10)


    mu_em_it = np.array(mu2[-1])[:, 0]
    gamma_em_it = np.diag(gamma2[-1])

    mu_ecme_it = np.array(mu1[-1])[:, 0]
    gamma_ecme_it = np.diag(gamma1[-1])

    mu_reml_it = mu3
    gamma_reml_it = gamma3

    mu_profile_it = mu4
    gamma_profile_it = gamma4

    mu_reml_ridge_it = mu5
    gamma_reml_ridge_it = gamma5


    return {
        "mu_em": mu_em_it,
        "gamma_em": gamma_em_it,
        "mu_ecme": mu_ecme_it,
        "gamma_ecme": gamma_ecme_it,
        "mu_reml": mu_reml_it,
        "gamma_reml": gamma_reml_it,
        "mu_profile": mu_profile_it,
        "gamma_profile": gamma_profile_it,
        "mu_reml_ridge": mu_reml_ridge_it,
        "gamma_reml_ridge": gamma_reml_ridge_it
    }


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(run_replication, range(1, n_rep + 1)))


    for i, res in enumerate(results):
        it = i + 1

        mu_em[it, :] = res["mu_em"]
        gamma_em[it, :] = res["gamma_em"]

        mu_ecme[it, :] = res["mu_ecme"]
        gamma_ecme[it, :] = res["gamma_ecme"]

        mu_reml[it, :] = res["mu_reml"]
        gamma_reml[it, :] = res["gamma_reml"]

        mu_profile[it, :] = res["mu_profile"]
        gamma_profile[it, :] = res["gamma_profile"]

        mu_reml_ridge[it, :] = res["mu_reml_ridge"]
        gamma_reml_ridge[it, :] = res["gamma_reml_ridge"]


    pd.DataFrame(mu_em).to_csv(f"./results_toycase_baseline/mu_EM_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_em).to_csv(f"./results_toycase_baseline/gamma_EM_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_ecme).to_csv(f"./results_toycase_baseline/mu_ECME_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_ecme).to_csv(f"./results_toycase_baseline/gamma_ECME_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_reml).to_csv(f"./results_toycase_baseline/mu_REML_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_reml).to_csv(f"./results_toycase_baseline/gamma_REML_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_profile).to_csv(f"./results_toycase_baseline/mu_profile_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_profile).to_csv(f"./results_toycase_baseline/gamma_profile_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_reml_ridge).to_csv(f"./results_toycase_baseline/mu_REML_ridge_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_reml_ridge).to_csv(f"./results_toycase_baseline/gamma_REML_ridge_nrep_{n_rep}_n_{n}.csv", index=False)
