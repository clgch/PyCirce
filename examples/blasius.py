import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import scipy.stats as st
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib import rc

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)
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

    mu_f = cp.CoolProp.PropsSI("V", "T", T, "P", P, "HEOS::Water")

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

    mu_f = cp.CoolProp.PropsSI("V", "T", T, "P", P, "HEOS::Water")

    theta = np.random.multivariate_normal(mu, gamma, size=n)

    return theta[:, 0] + theta[:, 1] * np.log10((G * D) / mu_f)


def compute_h(x):
    n = x.shape[0]

    P = x[:, 0]
    G = x[:, 1]
    T = x[:, 2]
    D = x[:, 3]

    mu_f = cp.CoolProp.PropsSI("V", "T", T, "P", P, "HEOS::Water")

    return np.repeat(1.0, n), np.log10((G * D) / mu_f)



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

n_rep = 1000

mu_em = np.zeros((n_rep + 1, 2))
gamma_em = np.zeros((n_rep + 1, 2))

mu_ecme = np.zeros((n_rep + 1, 2))
gamma_ecme = np.zeros((n_rep + 1, 2))

mu_reml = np.zeros((n_rep + 1, 2))
gamma_reml = np.zeros((n_rep + 1, 2))

mu_profile = np.zeros((n_rep + 1, 2))
gamma_profile = np.zeros((n_rep + 1, 2))

gamma_true = np.diag(np.array([(0.1)**2, 0.7]))
mu_true = np.array([np.log10(0.316) * 1.3, -0.25 * 1.3])

mu_em[0, :] = mu_true
gamma_em[0, :] = np.diag(gamma_true)

mu_ecme[0, :] = mu_true
gamma_ecme[0, :] = np.diag(gamma_true)

mu_reml[0, :] = mu_true
gamma_reml[0, :] = np.diag(gamma_true)

mu_profile[0, :] = mu_true
gamma_profile[0, :] = np.diag(gamma_true)


log_alpha_range = np.linspace(-3, 0, num=50)

bic_array = np.zeros(len(log_alpha_range))

n = 10

T = np.array([25, 110, 250])
P = np.array([60, 90, 110, 130])

data = np.zeros((n, 4))

data[:, 0] = P[np.random.choice(4, size=n, replace=True)] * 100000
data[:, 1] = np.random.uniform(250, 3000, size=n)

data[:, 2] = np.random.uniform(20, 40, size=n) * (
    data[:, 0] < 110 * 100000
) + np.random.uniform(20, 300, size=n) * (data[:, 0] > 100 * 100000)
data[:, 2] = data[:, 2] + 273.15

data[:, 3] = np.repeat(4e-3, n)

# Coverage computation using the linear Gaussian model
cov_rate = np.linspace(0.1, 1, 10)


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

    f_blasius_exp = generate_data_blasius(mu_true, gamma_true, data, it) + sig_eps

    h = np.zeros((2, n))
    h[:, :] = compute_h(data)

    #print(np.mean(h[1, :]) ** 2 / np.var(h[1, :]))

    bmin = 1/3.0 * np.amax((1) ** 2/(h[0, :] ** 2 + h[1, :] ** 2))


    circe_diag = pyc.CirceEMdiag(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_blasius_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000,
        tolerance=1e-4,
    )

    circe_diag_ecme = pyc.CirceECMEdiag(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_blasius_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000,
        tolerance=1e-4,
    )

    circe_reml = pyc.CirceREML(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_blasius_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000
    )


    circe_profile = pyc.CirceProfile(
        initial_mean=[np.log10(0.316), -0.25],
        initial_cov= bmin * np.identity(2),
        h=h,
        z_exp=f_blasius_exp,
        z_nom=np.repeat(0, n),
        sig_eps=sig_eps,
        niter=15000
    )


    mu1, gamma1, loglik, err_cov1, err_mean1 = circe_diag_ecme.estimate()
    #print(f"iterations ECME = {len(err_mean1)}/15000")

    mu2, gamma2, loglik2, err_cov2, err_mean2 = circe_diag.estimate()
    #print(f"iterations EM = {len(err_mean2)}/15000")

    mu3, gamma3, _ = circe_reml.estimate(n_starts=10)
    mu4, gamma4, _ = circe_profile.estimate(n_starts=10)


    mu_em_it = np.array(mu2[-1])[:, 0]
    gamma_em_it = np.diag(gamma2[-1])

    mu_ecme_it = np.array(mu1[-1])[:, 0]
    gamma_ecme_it = np.diag(gamma1[-1])

    mu_reml_it = mu3
    gamma_reml_it = gamma3

    mu_profile_it = mu4
    gamma_profile_it = gamma4

    # coverage_EM = np.zeros((len(cov_rate), 2))
    # coverage_ECME = np.zeros((len(cov_rate), 2))
    # coverage_REML = np.zeros((len(cov_rate), 2))
    # coverage_profile = np.zeros((len(cov_rate), 2))

    # interq_EM = np.zeros((len(cov_rate), 2))
    # interq_ECME = np.zeros((len(cov_rate), 2))
    # interq_REML = np.zeros((len(cov_rate), 2))
    # interq_profile = np.zeros((len(cov_rate), 2))

    # it_cov = 0
    # for alpha in cov_rate:

    #     q_min = st.norm().ppf(alpha/2)
    #     q_max = st.norm().ppf(1 - alpha/2)

    #     coverage_EM[it_cov, 1] = np.mean((f_blasius_exp > np.sum(h * mu_em_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_em_it[:, None] * h ** 2, axis=0) + 1)) * q_min) & (f_blasius_exp < np.sum(h * mu_em_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_em_it[:, None] * h ** 2, axis=0) + 1)) * q_max))
    #     coverage_EM[it_cov, 0] = 1 - alpha

    #     interq_EM[it_cov, 1] = np.median(np.sqrt(np.sum(gamma_em_it[:, None] * h ** 2, axis=0) + 1)) * (q_max - q_min)
    #     interq_EM[it_cov, 0] = 1 - alpha

    #     coverage_ECME[it_cov, 1] = np.mean((f_blasius_exp > np.sum(h * mu_ecme_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_ecme_it[:, None] * h ** 2, axis=0) + 1)) * q_min) & (f_blasius_exp < np.sum(h * mu_ecme_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_ecme_it[:, None] * h ** 2, axis=0) + 1)) * q_max))
    #     coverage_ECME[it_cov, 0] = 1 - alpha

    #     interq_ECME[it_cov, 1] = np.median(np.sqrt(np.sum(gamma_ecme_it[:, None] * h ** 2, axis=0) + 1)) * (q_max - q_min)
    #     interq_ECME[it_cov, 0] = 1 - alpha

    #     coverage_REML[it_cov, 1] = np.mean((f_blasius_exp > np.sum(h * mu_reml_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_reml_it[:, None] * h ** 2, axis=0) + 1)) * q_min) & (f_blasius_exp < np.sum(h * mu_reml_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_reml_it[:, None] * h ** 2, axis=0) + 1)) * q_max))
    #     coverage_REML[it_cov, 0] = 1 - alpha

    #     interq_REML[it_cov, 1] = np.median(np.sqrt(np.sum(gamma_reml_it[:, None] * h ** 2, axis=0) + 1)) * (q_max - q_min)
    #     interq_REML[it_cov, 0] = 1 - alpha

    #     coverage_profile[it_cov, 1] = np.mean((f_blasius_exp > np.sum(h * mu_profile_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_profile_it[:, None] * h ** 2, axis=0) + 1)) * q_min) & (f_blasius_exp < np.sum(h * mu_profile_it[:, None], axis=0) + (np.sqrt(np.sum(gamma_profile_it[:, None] * h ** 2, axis=0) + 1)) * q_max))
    #     coverage_profile[it_cov, 0] = 1 - alpha

    #     interq_profile[it_cov, 1] = np.median(np.sqrt(np.sum(gamma_profile_it[:, None] * h ** 2, axis=0) + 1)) * (q_max - q_min)
    #     interq_profile[it_cov, 0] = 1 - alpha

    #     it_cov += 1

    return {
        "mu_em": mu_em_it,
        "gamma_em": gamma_em_it,
        "mu_ecme": mu_ecme_it,
        "gamma_ecme": gamma_ecme_it,
        "mu_reml": mu_reml_it,
        "gamma_reml": gamma_reml_it,
        "mu_profile": mu_profile_it,
        "gamma_profile": gamma_profile_it,
        # "coverage_EM": coverage_EM,
        # "coverage_ECME": coverage_ECME,
        # "coverage_REML": coverage_REML,
        # "coverage_profile": coverage_profile,
        # "interq_EM": interq_EM,
        # "interq_ECME": interq_ECME,
        # "interq_REML": interq_REML,
        # "interq_profile": interq_profile,
    }


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_replication, range(1, n_rep + 1)))

    # coverage_EM_stack = np.zeros((n_rep, len(cov_rate)))
    # coverage_ECME_stack = np.zeros((n_rep, len(cov_rate)))
    # coverage_REML_stack = np.zeros((n_rep, len(cov_rate)))
    # coverage_profile_stack = np.zeros((n_rep, len(cov_rate)))

    # interq_EM_stack = np.zeros((n_rep, len(cov_rate)))
    # interq_ECME_stack = np.zeros((n_rep, len(cov_rate)))
    # interq_REML_stack = np.zeros((n_rep, len(cov_rate)))
    # interq_profile_stack = np.zeros((n_rep, len(cov_rate)))

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

    #     coverage_EM_stack[i, :] = res["coverage_EM"][:, 1]
    #     coverage_ECME_stack[i, :] = res["coverage_ECME"][:, 1]
    #     coverage_REML_stack[i, :] = res["coverage_REML"][:, 1]
    #     coverage_profile_stack[i, :] = res["coverage_profile"][:, 1]

    #     interq_EM_stack[i, :] = res["interq_EM"][:, 1]
    #     interq_ECME_stack[i, :] = res["interq_ECME"][:, 1]
    #     interq_REML_stack[i, :] = res["interq_REML"][:, 1]
    #     interq_profile_stack[i, :] = res["interq_profile"][:, 1]

    # # Coverage/interquantile diagnostics averaged (Monte-Carlo estimate) across
    # # all n_rep replications, one row per nominal confidence level in cov_rate.
    # coverage_EM = np.column_stack((1 - cov_rate, coverage_EM_stack.mean(axis=0)))
    # coverage_ECME = np.column_stack((1 - cov_rate, coverage_ECME_stack.mean(axis=0)))
    # coverage_REML = np.column_stack((1 - cov_rate, coverage_REML_stack.mean(axis=0)))
    # coverage_profile = np.column_stack((1 - cov_rate, coverage_profile_stack.mean(axis=0)))

    # interq_EM = np.column_stack((1 - cov_rate, interq_EM_stack.mean(axis=0)))
    # interq_ECME = np.column_stack((1 - cov_rate, interq_ECME_stack.mean(axis=0)))
    # interq_REML = np.column_stack((1 - cov_rate, interq_REML_stack.mean(axis=0)))
    # interq_profile = np.column_stack((1 - cov_rate, interq_profile_stack.mean(axis=0)))

    pd.DataFrame(mu_em).to_csv(f"./results/mu_EM_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_em).to_csv(f"./results/gamma_EM_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(coverage_EM).to_csv(f"./results/coverage_EM_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(interq_EM).to_csv(f"./results/interq_EM_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_ecme).to_csv(f"./results/mu_ECME_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_ecme).to_csv(f"./results/gamma_ECME_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(coverage_ECME).to_csv(f"./results/coverage_ECME_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(interq_ECME).to_csv(f"./results/interq_ECME_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_reml).to_csv(f"./results/mu_REML_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_reml).to_csv(f"./results/gamma_REML_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(coverage_REML).to_csv(f"./results/coverage_REML_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(interq_REML).to_csv(f"./results/interq_REML_nrep_{n_rep}_n_{n}.csv", index=False)

    pd.DataFrame(mu_profile).to_csv(f"./results/mu_profile_nrep_{n_rep}_n_{n}.csv", index=False)
    pd.DataFrame(gamma_profile).to_csv(f"./results/gamma_profile_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(coverage_profile).to_csv(f"./results/coverage_profile_nrep_{n_rep}_n_{n}.csv", index=False)
    #pd.DataFrame(interq_profile).to_csv(f"./results/interq_profile_nrep_{n_rep}_n_{n}.csv", index=False)


# sig_eps = np.random.normal(0, 4, n)
# f_blasius_exp = generate_data_blasius(mu_true, gamma_true, data, 0) + sig_eps

# for i in range(len(log_alpha_range)):

#     circe_ecme_ridge = pyc.CirceECMERidgediag(
#     initial_mean=[np.log10(0.316), -0.25],
#     initial_cov= bmin * np.identity(2),
#     h=h,
#     z_exp=f_blasius_exp,
#     z_nom=np.repeat(0, n),
#     sig_eps=sig_eps,
#     niter=15000,
#     tolerance=1e-4,
#     alpha=np.power(10, log_alpha_range[i])
#     )

#     mu4, gamma4, loglik4, err_cov4, err_mean4, bic = circe_ecme_ridge.estimate()

#     bic_array[i] = bic

#     print(bic)
#     bic_array[i] = bic

# bic_diff = np.diff(bic_array)

# plt.plot(log_alpha_range, bic_array, '*')
# plt.xlabel(r"$\log(\alpha)$", fontsize=20)
# plt.ylabel(r"BIC", fontsize=20)
# plt.show()
