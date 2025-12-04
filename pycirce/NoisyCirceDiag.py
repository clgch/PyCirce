#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CEA/DES 2025

@author: Clément GAUCHY
"""
import numpy as np
import copy


class NoisyCirceDiag:
    """
    Expectation Maximisation (EM) algorithm for probabilistic inversion of thermohydraulic correlation, with diagonal covariance matrix.
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
            raise ValueError("Please provide a Jacobian matrix h of size (n_parameters, n_observations) of the thermohydraulic numerical simulator")
        if z_nom is None:
            raise ValueError("Please provide a vector of nominal simulations in z_nom")
        if sig_eps is None:
            raise ValueError("Please provide a vector of measurement standard deviations in sig_eps")

        # Check dataset size consistency
        if not (h.shape[2] == len(z_exp) == len(z_nom)):
            raise ValueError("ERROR: size inconsistencies between the Jacobian h, the vector z_exp and z_nom !")

        # Set default initial mu and cov
        if initial_mean is None:
            self.initial_mean = np.ones(size=h.shape[0]).reshape(-1, 1)
        else: 
            self.initial_mean = np.array(initial_mean).reshape(-1, 1)

        if initial_cov is None:
            self.initial_cov = np.diag(np.ones(size=h.shape[0]))
        else:
            self.initial_cov = initial_cov
        
        # Set default tolerance 
        if tolerance is None:
            self.tolerance = 1e-5
        else: 
            self.tolerance = tolerance
        
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
        Compute one iteration of the EM iterative algorithm and returns the updated mean and covariance
        """

        # Residual for each observation: z_exp - z_nom - h_i^T mean
        resid = z_exp - z_nom - np.sum(h * mean[:, None, None], axis=0) # (N_MC, n)
        
        # Denominators: h_i^T cov h_i + sig_eps^2, with diagonal cov
        denom = sig_eps ** 2 + np.sum(gamma[:, None, None] * h ** 2, axis=0) # (N_MC, n)

        # g_ji = (cov @ h_i)_j = gamma_j * h_ji
        g = gamma[:, None, None] * h  # (p, N_MC, n)

        # b_ij and S_ij for diagonal update
        b_MC = g * resid[None, :] / denom[None, :] # (p, N_MC, n)
        S_diag = (g ** 2) / denom[None, :]  # (p, N_Mc, n)

        # Empirical mean on both the observations and the Monte-Carlo sample
        delta_cov_diag_MC = b_MC ** 2 - S_diag
        delta_cov_diag = np.mean(delta_cov_diag_MC, axis=(-2, -1)) # (p,)
        delta_mean = np.mean(b_MC, axis=(-2, -1)) # (p,)

        # Empirical mean on the Monte-Carlo sample ONLY
        delta_mean_sq = np.mean(b_MC, axis=2) ** 2 # (p, n)
        
        # Parameters update
        gamma_new = gamma + delta_cov_diag - np.mean(delta_mean_sq, axis=1)
        mean_new = mean + delta_mean

        return mean_new, gamma_new


    def estimate(self):
        n = len(self.z_exp)
        iterator = 0

        mean = self._mean0.copy()
        gamma = self._gamma0.copy()

        cov_list = [np.diag(gamma)]
        mean_list = [mean.reshape(-1, 1)]

        mean, gamma = self._one_step_em_diag(
            mean, gamma, self.h, self.z_exp, self.z_nom, self.sig_eps
        )

        cov_list.append(np.diag(gamma))
        mean_list.append(mean.reshape(-1, 1))

        rel_diff_cov = np.abs(cov_list[-1].diagonal() - cov_list[-2].diagonal()) / np.abs(
            cov_list[-1].diagonal()
        )
        err_cov = float(np.max(rel_diff_cov))

        rel_diff_mean = np.abs(mean_list[-1] - mean_list[-2]) / np.abs(mean_list[-1])
        err_mean = float(np.max(rel_diff_mean))

        err_cov_list = [err_cov]
        err_mean_list = [err_mean]

        while (err_cov > self.tolerance or err_mean > self.tolerance) and iterator < self.niter :
            

            mean, gamma = self._one_step_em_diag(
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
        
        return mean_list, cov_list, err_cov_list, err_mean_list










