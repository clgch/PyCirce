import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 22})
rc("lines", linewidth=3)

rougeCEA = "#b81420"
bleuEDF = "#10367a"


def relative_rmse(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Compute relative RMSE between each row of `est` and the true vector,
    in the sense of the L2 norm of (est_i - true) / true.

    est  : shape (n_samples, p)
    true : shape (p,)

    Returns
    -------
    rrmse : shape (n_samples,)
        For each row i, sqrt( mean_j( ((est_ij - true_j) / true_j)^2 ) ).
    """
    rel = (est - true[None, :]) / true[None, :]  # (n_samples, p)
    mse = np.abs(rel).mean(axis=1)
    return mse


def main():
    # Project root is the parent directory of this 'examples' folder
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")

    # True parameters (must match those in blasius.py)
    gamma_true = np.diag(np.array([0.1, 0.1]))
    mu_true = np.array([np.log10(0.316), -0.25])
    gamma_true_diag = np.diag(gamma_true)

    # Load results (filenames are those written in blasius.py)
    mu_em = pd.read_csv(os.path.join(results_dir, "mu_em_nrep_200.csv")).values
    gamma_em = pd.read_csv(os.path.join(results_dir, "gamma_em_nrep_200.csv")).values
    mu_ecme = pd.read_csv(os.path.join(results_dir, "mu_ecme_nrep_200.csv")).values
    gamma_ecme = pd.read_csv(
        os.path.join(results_dir, "gamma_ecme_nrep_200.csv")
    ).values

    # Drop the first row (it contains the true values, see blasius.py)
    mu_em_est = mu_em[1:, :]
    gamma_em_est = gamma_em[1:, :]
    mu_ecme_est = mu_ecme[1:, :]
    gamma_ecme_est = gamma_ecme[1:, :]

    # Relative RMSE for each replication
    rrmse_mu_em = relative_rmse(mu_em_est, mu_true)
    rrmse_mu_ecme = relative_rmse(mu_ecme_est, mu_true)
    rrmse_gamma_em = relative_rmse(gamma_em_est, gamma_true_diag)
    rrmse_gamma_ecme = relative_rmse(gamma_ecme_est, gamma_true_diag)

    # One figure with box plots comparing EM vs ECME
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    # Positions for the two methods on x-axis
    positions = [1, 2]
    labels = ["EM", "ECME"]

    # Mean (mu) relative RMSE
    ax = axes[0]
    boxplot_data = ax.boxplot(
        [rrmse_mu_em, rrmse_mu_ecme],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
    )
    for box, color in zip(boxplot_data["boxes"], [bleuEDF, rougeCEA]):
        box.set_facecolor(color)
        box.set_alpha(0.6)
    ax.set_ylabel(r"Relative MAE on $\mu$")
    ax.set_title(r"Relative MAE of $\mu$ estimates")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)

    # Gamma (variance) relative RMSE
    ax = axes[1]
    boxplot_data = ax.boxplot(
        [rrmse_gamma_em, rrmse_gamma_ecme],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
    )
    for box, color in zip(boxplot_data["boxes"], [bleuEDF, rougeCEA]):
        box.set_facecolor(color)
        box.set_alpha(0.6)
    ax.set_ylabel(r"Relative MAE on $\gamma$")
    ax.set_title(r"Relative MAE of $\gamma$ estimates")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "blasius_relative_rmse_boxplots.png")
    plt.savefig(out_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
