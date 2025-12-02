#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CEA/DES 2025

@author: Clément GAUCHY
"""
import numpy as np
import copy


class CirceECMEdiag:
    """
    Expectation Conditional Maximisation Either (ECME) algorithm for probabilistic inversion of thermohydraulic correlation, with diagonal covariance matrix.
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
        if not (h.shape[1] == len(z_exp) == len(z_nom)):
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


    @staticmethod
    def _one_step_ecme(mean, cov, h, z_exp, z_nom, sig_eps, n):
        """
        Compute one iteration of the ECME iterative algorithm and returns the updated mean and covariance
        """
        b = []
        S = []
        delta_cov = []
        H_tilde_list = []

        for i in range(n):
            h_i = h[:, i].reshape(-1, 1)

            b.append(cov @ (h_i * ((z_exp[i] - z_nom[i]) - h_i.T @ mean)) / (h_i.T @ cov @ h_i + sig_eps[i] ** 2))

            S.append(cov @ h_i @ h_i.T @ cov / (h_i.T @ cov @ h_i + sig_eps[i] ** 2))

            delta_cov.append(b[-1] @ b[-1].T - S[-1])
        
        cov_new = np.diag(np.diag(copy.deepcopy(cov) + np.mean(delta_cov, axis=0)))  

        #print(f"cov_new = {cov_new}")

        for i in range(n):
            h_i = h[:, i].reshape(-1, 1)
            H_tilde_list.append(h_i @ h_i.T /(h_i.T @ cov_new @ h_i + sig_eps[i] ** 2))
        
        H_tilde = np.sum(H_tilde_list, axis=0)
        #H_tilde = 0.5 * (H_tilde + H_tilde.T) 

        #L = np.linalg.cholesky(H_tilde)
        #L_inv = np.linalg.inv(L)
        #H_tilde_inv = L_inv.T @ L_inv

        ### Computation of the Moore Penrose inverse of H_tilde ###

        # Compute the SVD of H_tilde
        U, s, Vt = np.linalg.svd(H_tilde, full_matrices=False)

        # Compute the pseudoinverse of the diagonal matrix s
        s_pseudo = np.zeros_like(s)

        non_zero = s > 1e-10  # Threshold for non-zero singular values
        s_pseudo[non_zero] = 1 / s[non_zero]

        # Compute Moore Penrose inverse
        H_plus = Vt.T @ np.diag(s_pseudo) @ U.T

        mean_new = 0
        for i in range(n):
            h_i = h[:, i]
            mean_new += H_plus @ h_i * (z_exp[i] - z_nom[i]) / (h_i.T @ cov_new @ h_i + sig_eps[i] ** 2) 
            #mean_new += H_tilde_inv @ h_i * (z_exp[i] - z_nom[i])

        #print(f"mean_new = {mean_new}")
        return mean_new.reshape(-1, 1), cov_new
     
    @staticmethod
    def _loglik(mean, cov, h, z_exp, z_nom, sig_eps, n):
        loglik = 0

        for i in range(n):
            z_prime = z_exp[i] - z_nom[i]
            h_i = h[:, i]

            loglik -= 0.5 * (z_prime - h_i.T @ mean) ** 2 / (h_i.T @ cov @ h_i + sig_eps[i] ** 2) + 0.5 * np.log(h_i.T @ cov @ h_i + sig_eps[i] ** 2)
        
        return loglik


    def estimate(self):
        n = len(self.z_exp)
        iterator = 0 

        cov_list = [self.initial_cov]
        mean_list = [self.initial_mean]

        loglik_list = [self._loglik(self.initial_mean, self.initial_cov, self.h, self.z_exp, self.z_nom, self.sig_eps, n)]  

        mean_new, cov_new = self._one_step_ecme(self.initial_mean, self.initial_cov, self.h, self.z_exp, self.z_nom, self.sig_eps, n)

        cov_list += [cov_new]
        mean_list += [mean_new]

        #err_cov = np.linalg.norm(cov_list[-1] - cov_list[-2])/np.linalg.norm(cov_list[-2])
        #err_mean = np.linalg.norm(mean_list[-1] - mean_list[-2])/np.linalg.norm(mean_list[-2])

        rel_diff = np.abs(np.diag(cov_list[-1]) - np.diag(cov_list[-2])) / np.abs(np.diag(cov_list[-1]))
        err_cov = np.max(rel_diff)

        rel_diff = np.abs(mean_list[-1] - mean_list[-2]) / np.abs(mean_list[-1])
        err_mean = np.max(rel_diff)

        err_cov_list = [err_cov]
        err_mean_list = [err_mean]

        # while (err_cov > self.tolerance or err_mean > self.tolerance) and iterator < self.niter :
        #     mean_new, cov_new = self._one_step_ecme(mean_list[-1], cov_list[-1], self.h, self.z_exp, self.z_nom, self.sig_eps, n)

        while (err_cov > self.tolerance or err_mean > self.tolerance) and iterator < self.niter :
            
            loglik_list += [self._loglik(mean_list[-1], cov_list[-1], self.h, self.z_exp, self.z_nom, self.sig_eps, n)] 
            
            mean_new, cov_new = self._one_step_ecme(mean_list[-1], cov_list[-1], self.h, self.z_exp, self.z_nom, self.sig_eps, n)
            
            cov_list += [cov_new]
            mean_list += [mean_new]

            rel_diff = np.abs(np.diag(cov_list[-1]) - np.diag(cov_list[-2])) / np.abs(np.diag(cov_list[-1]))
            err_cov = np.max(rel_diff)

            rel_diff = np.abs(mean_list[-1] - mean_list[-2]) / np.abs(mean_list[-1])
            err_mean = np.max(rel_diff)

            err_cov_list += [err_cov]
            err_mean_list += [err_mean]

            iterator += 1
            #print(iterator)
            #print(f"err cov = {err_cov}")
            #print(f"err mean = {err_mean}")
        
        return mean_list, cov_list, loglik_list, err_cov_list, err_mean_list










