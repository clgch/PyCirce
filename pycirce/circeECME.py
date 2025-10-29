#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CEA 2025

@author: Clément GAUCHY
"""
import numpy as np
import copy


class CirceECME:
    """
    Expectation Conditional Maximisation Either (ECME) algorithm for probabilistic inversion of thermohydraulic correlation.
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
    if not (h.shape[1] == len(z_exp) == len(z_nom) == len(sig_eps)):
        raise ValueError("ERROR: size inconsistencies between the Jacobian h, the vector z_exp and z_nom !")

    # Set default initial mu and cov
    if initial_mean is None:
        self.initial_mean = np.ones(size=h.shape[0])
    if initial_cov is None:
        self.initial_cov = np.diag(np.ones(size=h.shape[0]))
    
    # Set default tolerance 
    if tolerance is None:
        self.tolerance = 1e-5

    @staticmethod
    def one_step_ecme(mean, cov, h, z_exp, z_nom, n):
        """
        Compute one iteration of the ECME iterative algorithm and returns the updated mean and covariance
        """
        b = []
        S = []
        delta_cov = []
        H_tilde_list = []

        for i in range(n):
            h_i = self.h[:, i]

            b.append(cov @ (h_i * ((z_exp[i] - z_nom[i]) - h_i.dot(mean))) / (h_i.dot(cov @ hi) + self.sig_eps[i] ** 2))

            S.append(cov @ hi @ hi.T @ cov / ((h_i.dot(cov @ hi) + self.sig_eps[i] ** 2)))

            delta_cov.append(b[-1] @ b[-1].T - S[-1])
        
        cov_new = copy.deepcopy(cov) + np.mean(delta_cov)  

        for i in range(n):
            H_tilde_list.append(h_i @ h_i.T /((h_i.dot(cov_list[-1] @ hi) + self.sig_eps[i] ** 2)))
        
        H_tilde = np.sum(H_tilde_list)
        H_tilde = 0.5 * (H_tilde + H_tilde.T)

        L = np.linalg.cholesky(H_tilde)
        L_inv = np.linalg.inv(L)
        H_tilde_inv = L_inv.T @ L_inv

        mean_new = 0
        for i in range(n):
            h_i = self.h[:, i]
            mean_new += H_tilde_inv @ h_i * (z_exp[i] - z_nom[i]) / ((h_i.dot(cov_list[-1] @ hi) + self.sig_eps[i] ** 2)) 

        return mean_new, cov_new
     


    def estimate(self):
        n = len(z_exp)
        iterator = 0 

        cov_list = [initial_cov]
        mean_list = [initial_mean]

        mean_new, cov_new = one_step_ecme(initial_mean, initial_cov, self.h, self.z_exp, self.z_nom, n)

        cov_list += [cov_new]
        mean_list += [mean_new]

        err = np.linalg.norm(cov_list[-1] - cov_list[-2])




