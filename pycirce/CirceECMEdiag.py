#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal-covariance ECME algorithm for probabilistic inversion of
thermohydraulic correlations (pure Python, vectorised NumPy).
"""

import numpy as np


class CirceECMEdiag:
    """
    Expectation Conditional Maximisation Either (ECME) algorithm for
    probabilistic inversion of thermohydraulic correlation, with
    diagonal covariance matrix.
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

        # Cast to NumPy arrays
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

        # Initial mean / covariance (full diagonal matrix for public API)
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

        # Internal diagonal representation
        self._mean0 = self.initial_mean.reshape(-1).astype(float)
        self._gamma0 = np.diag(self.initial_cov).astype(float)

    @staticmethod
    def _one_step_ecme_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        One ECME step using only the diagonal of the covariance (gamma).

        Parameters
        ----------
        mean  : (p,)
        gamma : (p,)
        h     : (p, n)
        """
        p, n = h.shape

        # E-step: compute b_i and S_i as in EM, but maintain gamma as diagonal
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)
        denom = sig_eps ** 2 + np.sum(gamma[:, None] * h ** 2, axis=0)  # (n,)

        g = gamma[:, None] * h  # (p, n)
        b = g * resid[None, :] / denom[None, :]  # (p, n)
        S_diag = (g ** 2) / denom[None, :]  # (p, n)
        delta_cov_diag_mat = b ** 2 - S_diag  # (p, n)

        delta_cov_diag = np.mean(delta_cov_diag_mat, axis=1)  # (p,)
        gamma_new = gamma + delta_cov_diag  # ECME: no -delta_mean^2 term

        # M-step for mean (exact maximisation):
        # Build H_tilde and its pseudoinverse with diagonal gamma_new
        # H_tilde = sum_i (h_i h_i^T / (h_i^T cov_new h_i + sig_eps_i^2))
        # with cov_new diagonal(gamma_new)
        w = 1.0 / (
            sig_eps ** 2 + np.sum(gamma_new[:, None] * h ** 2, axis=0)
        )  # (n,)

        # Weighted outer products: sum_i w_i * h_i h_i^T
        # h has shape (p, n); apply weights along the observation axis
        H_tilde = (h * w[None, :]) @ h.T  # (p, p)

        # Moore–Penrose inverse of H_tilde via SVD
        U, s, Vt = np.linalg.svd(H_tilde, full_matrices=False)
        s_pseudo = np.zeros_like(s)
        non_zero = s > 1e-10
        s_pseudo[non_zero] = 1.0 / s[non_zero]
        H_plus = Vt.T @ np.diag(s_pseudo) @ U.T

        # Right-hand side: sum_i h_i * (z_exp[i] - z_nom[i]) / denom_i
        rhs = h @ ((z_exp - z_nom) * w)  # (p,)

        mean_new = H_plus @ rhs  # (p,)

        return mean_new, gamma_new

    @staticmethod
    def _loglik_diag(mean, gamma, h, z_exp, z_nom, sig_eps):
        """
        Observed-data log-likelihood with diagonal covariance.
        """
        resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)  # (n,)
        denom = sig_eps ** 2 + np.sum(gamma[:, None] * h ** 2, axis=0)  # (n,)
        loglik = -0.5 * np.sum(resid ** 2 / denom) + 0.5 * np.sum(np.log(denom))
        return loglik

    def estimate(self):
        """
        Run the ECME iterations until convergence (or until niter is reached).

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
        n = len(self.z_exp)
        iterator = 0

        mean = self._mean0.copy()
        gamma = self._gamma0.copy()

        cov_list = [np.diag(gamma)]
        mean_list = [mean.reshape(-1, 1)]

        loglik_list = [
            self._loglik_diag(mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps)
        ]

        mean, gamma = self._one_step_ecme_diag(
            mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
        )

        cov_list.append(np.diag(gamma))
        mean_list.append(mean.reshape(-1, 1))
        loglik_list.append(
            self._loglik_diag(mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps)
        )

        rel_diff_cov = np.abs(cov_list[-1].diagonal() - cov_list[-2].diagonal()) / np.abs(
            cov_list[-1].diagonal()
        )
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
                )
            )

            mean, gamma = self._one_step_ecme_diag(
                mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
            )

            cov_list.append(np.diag(gamma))
            mean_list.append(mean.reshape(-1, 1))

            rel_diff_cov = np.abs(
                cov_list[-1].diagonal() - cov_list[-2].diagonal()
            ) / np.abs(cov_list[-1].diagonal())
            err_cov = float(np.max(rel_diff_cov))

            rel_diff_mean = np.abs(mean_list[-1] - mean_list[-2]) / np.abs(mean_list[-1])
            err_mean = float(np.max(rel_diff_mean))

            err_cov_list.append(err_cov)
            err_mean_list.append(err_mean)

            iterator += 1

        return mean_list, cov_list, loglik_list, err_cov_list, err_mean_list










