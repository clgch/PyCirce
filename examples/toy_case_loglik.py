import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"


def _loglik(mean, cov, h, z_exp, z_nom, sig_eps, n):
    loglik = 0

    for i in range(n):
        z_prime = z_exp[i] - z_nom[i]
        h_i = h[:, i]

        loglik -= 0.5 * (z_prime - h_i.T @ mean) ** 2 / (
            h_i.T @ cov @ h_i + sig_eps[i] ** 2
        ) + 0.5 * np.log(h_i.T @ cov @ h_i + sig_eps[i] ** 2)

    return loglik


np.random.seed(10)

p = 2
n = 30
niter = 1000

mu_true = np.array([55, 30])

rho = 0.2
v1 = 2 * (7.5) ** 2
v2 = 2 * (1.5) ** 2
gamma_true = np.array([[v1, rho * np.sqrt(v1 * v2)], [rho * np.sqrt(v1 * v2), v2]])

lambda_sample = np.random.multivariate_normal(mu_true, gamma_true, size=n)
X = np.random.normal(size=(p, n))
epsilon = np.random.normal(loc=0, scale=0.1, size=n)

z_exp = np.zeros(n)
z_nom = np.zeros(n)
for i in range(n):
    z_exp[i] = np.sum(X.T[i] * lambda_sample[i]) + epsilon[i]
    z_nom[i] = 0


def f(x1, x2, x3, x4, x5):
    mean = np.array([x1, x2]).reshape(-1, 1)
    cov = np.array([[x3, x5], [x5, x4]])

    return _loglik(mean, cov, X, z_exp, z_nom, sig_eps=np.repeat(0.1, n), n=n)


# Define the fixed values for each variable when not in the moving pair
fixed_values = {
    0: 55,  # x1
    1: 30,  # x2
    2: 2 * (7.5) ** 2,  # x3
    3: 2 * (1.5) ** 2,  # x4
    4: rho * np.sqrt(v1 * v2),  # x5
}

# Define the range for the two moving variables
x_range = np.linspace(0.1, 1.9, 100)

# List of variable names for clarity
var_names = ["$m_1$", "$m_2$", "$\sigma^2_1$", "$\sigma^2_2$", "$\sigma_{12}$"]

# Iterate over all possible pairs of variables
for pair in combinations(range(5), 2):
    i, j = pair
    # Create a grid for the two moving variables
    x_moving1 = x_range * fixed_values[i]
    x_moving2 = x_range * fixed_values[j]
    X1, X2 = np.meshgrid(x_moving1, x_moving2)

    # Prepare the fixed values for the other variables
    def f_pair(x_i, x_j):
        args = [0] * 5
        args[i] = x_i
        args[j] = x_j
        for k in range(5):
            if k != i and k != j:
                args[k] = fixed_values[k]
        return f(*args)

    # Vectorize the function for the grid
    f_pair_vec = np.vectorize(f_pair)
    Z = f_pair_vec(X1, X2)

    # Plot the cross-cut
    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z,
        extent=[x_moving1.min(), x_moving1.max(), x_moving2.min(), x_moving2.max()],
        origin="lower",
        cmap="inferno",
        aspect="auto",
    )
    plt.colorbar(label="Log-likelihood")
    plt.xlabel(rf"{var_names[i]}")
    plt.ylabel(rf"{var_names[j]}")
    plt.show()
