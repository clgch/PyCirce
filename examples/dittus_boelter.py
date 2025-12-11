import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import random
import jax.numpy as jnp
from kernax.kernels import Energy
from kernax import KernelHerding
from matplotlib import rc

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"


def dittus_boelter_corr(x):
    """
    Helper function that computes the Nusselt number using the Dittus Boelter correlation
    """

    P = x[:, 0]
    G = x[:, 1]
    Tp = x[:, 2]
    D = x[:, 3]

    Tsat = cp.CoolProp.PropsSI("T", "P", P, "Q", 0, "HEOS::Water")
    Tf = (Tsat + Tp) / 2

    lambda_f = cp.CoolProp.PropsSI("L", "T", Tf, "P", P, "HEOS::Water") * 1e-3
    mu_f = cp.CoolProp.PropsSI("V", "T", Tf, "P", P, "HEOS::Water")
    c_p = cp.CoolProp.PropsSI("C", "T", Tf, "P", P, "HEOS::Water")

    # return 0.023 * (lambda_f/ D)  * (G * D / mu_f) ** 0.8 * (c_p * mu_f / lambda_f) ** 0.4
    return (
        np.log10(0.023)
        + np.log10(lambda_f / D)
        + np.log10(G * D / mu_f) * 0.8
        + np.log10(c_p * mu_f / lambda_f) * 0.4
    )


def generate_data_becker(data, seed):
    """
    Generate postCHF data according to Becker's experimental dataset
    """
    np.random.seed(seed)

    n = data.shape[0]

    P = data[:, 0]
    G = data[:, 1]
    Tp = data[:, 2]
    D = data[:, 3]

    Tsat = cp.CoolProp.PropsSI("T", "P", P, "Q", 0, "HEOS::Water")
    Tf = (Tsat + Tp) / 2

    lambda_f = cp.CoolProp.PropsSI("L", "T", Tf, "P", P, "HEOS::Water") * 1e-3
    mu_f = cp.CoolProp.PropsSI("V", "T", Tf, "P", P, "HEOS::Water")
    c_p = cp.CoolProp.PropsSI("C", "T", Tf, "P", P, "HEOS::Water")

    gamma_true = np.diag(np.array([0.1, 0.1, 0.1]))
    mu_true = np.array([1, 0.8, 0.4])

    theta = np.random.multivariate_normal(mu_true, gamma_true, size=n)

    # return 0.023 * np.power((lambda_f / D), theta[:, 0])  * np.power((G * D / mu_f), theta[:, 1]) * np.power((c_p * mu_f / lambda_f), theta[:, 2])
    return (
        np.log10(0.023)
        + np.log10(lambda_f / D) * theta[:, 0]
        + np.log10(G * D / mu_f) * theta[:, 1]
        + np.log10(c_p * mu_f / lambda_f) * theta[:, 2]
    )


def compute_h(x):
    P = x[:, 0]
    G = x[:, 1]
    Tp = x[:, 2]
    D = x[:, 3]

    Tsat = cp.CoolProp.PropsSI("T", "P", P, "Q", 0, "HEOS::Water")
    Tf = (Tsat + Tp) / 2

    lambda_f = cp.CoolProp.PropsSI("L", "T", Tf, "P", P, "HEOS::Water") * 1e-3
    mu_f = cp.CoolProp.PropsSI("V", "T", Tf, "P", P, "HEOS::Water")
    c_p = cp.CoolProp.PropsSI("C", "T", Tf, "P", P, "HEOS::Water")

    return (
        np.log10(lambda_f / D),
        np.log10((G * D / mu_f)),
        np.log10((c_p * mu_f / lambda_f)),
    )


np.random.seed(0)
random.seed()

# Perform kernel herding (support points) to select 20 inputs variables from the 315-sized Becker's dataset

n = 50

X = np.zeros((315, 4))
X[:, :3] = pd.read_csv("./examples/becker_design.csv").values[:, 1:]
X[:, 3] = np.repeat(8e-3, 315)

jax_X = jnp.array(X)

quantization_fn = KernelHerding(jax_X, kernel_fn=Energy)
idx = quantization_fn(m=n)

X_sp = X[idx, :]

sig_eps = np.random.normal(0, 0.01, n) * dittus_boelter_corr(X_sp)
z_sim = generate_data_becker(X_sp, 0) + sig_eps

h = dittus_boelter_corr(X_sp)

# plt.plot(h, z_sim, '*')
# plt.show()

h = np.zeros((3, n))
h[:, :] = compute_h(X_sp)

circe_diag = pyc.CirceEMdiag(
    initial_mean=[1, 0.8, 0.4],
    initial_cov=np.identity(3),
    h=h,
    z_exp=z_sim,
    z_nom=np.repeat(np.log10(0.023), n),
    sig_eps=sig_eps,
    niter=1000,
)

mu1, gamma1, loglik, err_cov1, err_mean1 = circe_diag.estimate()


print(f"iterations = {len(err_mean1)}")
print(f"err mean (EM CIRCE) = {err_mean1[-1]}")
print(f"err cov (EM CIRCE) = {err_cov1[-1]}")

print(f"mu (EM CIRCE) = {mu1[-1]}")
print(f"gamma (EM CIRCE) = {np.diag(gamma1[-1])}")
