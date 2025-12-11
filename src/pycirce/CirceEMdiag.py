#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal-covariance EM algorithm for probabilistic inversion of
thermohydraulic correlations (pure Python, vectorised NumPy).
"""

import numpy as np


class CirceEMdiag:
    """
    Expectation Maximisation (EM) algorithm for probabilistic inversion of
    thermohydraulic correlation, with diagonal covariance matrix.
    """

    def __init__(
        self,
        initial_mean=None,
        initial_cov=None,
        tolerance=None,
        h=None,
        z_exp=None,
        z_nom=None,
        sig_eps=None,
        niter=None,
    ):
        # Missing parameters
        if z_exp is None:
            raise ValueError("Please provide a vector of experimental values in z_exp")
        if h is None:
            raise ValueError(
                "Please provide a Jacobian matrix h of size "
                "(n_parameters, n_observations) of the thermohydraulic numerical simulator"
            )
        if z_nom is None:
            raise ValueError("Please provide a vector of nominal simulations in z_nom")
        if sig_eps is None:
            raise ValueError(
                "Please provide a vector of measurement standard deviations in sig_eps"
            )

        # Cast to NumPy arrays for consistency
        h = np.asarray(h, dtype=float)
        z_exp = np.asarray(z_exp, dtype=float)
        z_nom = np.asarray(z_nom, dtype=float)
        sig_eps = np.asarray(sig_eps, dtype=float)

        # Check dataset size consistency
        if not (h.shape[1] == len(z_exp) == len(z_nom) == len(sig_eps)):
            raise ValueError(
                "ERROR: size inconsistencies between the Jacobian h, "
                "the vectors z_exp, z_nom and sig_eps!"
            )

        p = h.shape[0]

        # Initial mean and covariance (full diagonal matrix for public API)
        if initial_mean is None:
            self.initial_mean = np.ones(p).reshape(-1, 1)
        else:
            self.initial_mean = np.array(initial_mean, dtype=float).reshape(-1, 1)

        if initial_cov is None:
            self.initial_cov = np.diag(np.ones(p, dtype=float))
        else:
            self.initial_cov = np.array(initial_cov, dtype=float)

        # Tolerance and iteration limit
        self.tolerance = 1e-5 if tolerance is None else float(tolerance)

        self.h = h
        self.z_exp = z_exp
        self.z_nom = z_nom
        self.sig_eps = sig_eps
        self.niter = niter

        # Internal 1D representations (diagonal covariance as a vector)
        self._mean0 = self.initial_mean.reshape(-1).astype(float)
        self._gamma0 = np.diag(self.initial_cov).astype(float)

    @staticmethod
    def _one_step_em_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        One EM step using only the diagonal of the covariance (gamma).

        Parameters
        ----------
        mean  : (p,)
        gamma : (p,)
        h     : (p, n)
        """

        # Residual for each observation: z_exp - z_nom - h_i^T mean
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)

        # Denominators: h_i^T cov h_i + sig_eps^2, with diagonal cov
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,)

        # g_ji = (cov @ h_i)_j = gamma_j * h_ji
        g = gamma[:, None] * h  # (p, n)

        # b_ij and S_ij for diagonal update
        b = g * resid[None, :] / denom[None, :]  # (p, n)
        S_diag = (g**2) / denom[None, :]  # (p, n)
        delta_cov_diag_mat = b**2 - S_diag  # (p, n)

        # Empirical means over observations
        delta_mean = np.mean(b, axis=1)  # (p,)
        delta_cov_diag = np.mean(delta_cov_diag_mat, axis=1)  # (p,)

        mean_new = mean + delta_mean
        gamma_new = gamma + delta_cov_diag - delta_mean**2

        return mean_new, gamma_new

    @staticmethod
    def _loglik_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        Log-likelihood with diagonal covariance (gamma).
        """
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)
        denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0)  # (n,)

        loglik = -0.5 * np.sum(resid**2 / denom) + 0.5 * np.sum(np.log(denom))
        return np.array([loglik])

    def estimate(self):
        """
        Run the EM iterations until convergence (or until niter is reached).

        Returns
        -------
        mean_list : list of ndarray
            Sequence of mean vectors.
        cov_list : list of ndarray
            Sequence of (diagonal) covariance matrices.
        loglik_list : list of float
            Log-likelihood values at each iteration.
        err_cov_list : list of float
            Max relative change of the covariance diagonal at each iteration.
        err_mean_list : list of float
            Max relative change of the mean vector at each iteration.
        """
        iterator = 0

        mean = self._mean0.copy()
        gamma = self._gamma0.copy()

        cov_list = [np.diag(gamma)]
        mean_list = [mean.reshape(-1, 1)]

        loglik_list = [
            self._loglik_diag(
                mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
            )[0]
        ]

        mean, gamma = self._one_step_em_diag(
            mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
        )

        cov_list.append(np.diag(gamma))
        mean_list.append(mean.reshape(-1, 1))
        loglik_list.append(
            self._loglik_diag(
                mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
            )[0]
        )

        rel_diff_cov = np.abs(
            cov_list[-1].diagonal() - cov_list[-2].diagonal()
        ) / np.abs(cov_list[-1].diagonal())
        err_cov = float(np.max(rel_diff_cov))

        rel_diff_mean = np.abs(mean_list[-1] - mean_list[-2]) / np.abs(mean_list[-1])
        err_mean = float(np.max(rel_diff_mean))

        err_cov_list = [err_cov]
        err_mean_list = [err_mean]

        while (err_cov > self.tolerance or err_mean > self.tolerance) and (
            self.niter is None or iterator < self.niter
        ):
            loglik_list.append(
                self._loglik_diag(
                    mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
                )[0]
            )

            mean, gamma = self._one_step_em_diag(
                mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
            )

            cov_list.append(np.diag(gamma))
            mean_list.append(mean.reshape(-1, 1))

            rel_diff_cov = np.abs(
                cov_list[-1].diagonal() - cov_list[-2].diagonal()
            ) / np.abs(cov_list[-1].diagonal())
            err_cov = float(np.max(rel_diff_cov))

            rel_diff_mean = np.abs(mean_list[-1] - mean_list[-2]) / np.abs(
                mean_list[-1]
            )
            err_mean = float(np.max(rel_diff_mean))

            err_cov_list.append(err_cov)
            err_mean_list.append(err_mean)

            iterator += 1

        return mean_list, cov_list, loglik_list, err_cov_list, err_mean_list
