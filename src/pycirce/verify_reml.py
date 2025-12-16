import sys
import os
import numpy as np

# Add 'src' to path so we can import pycirce package
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/pycirce
src_dir = os.path.dirname(current_dir) # src
sys.path.insert(0, src_dir)

from pycirce.CirceREML import CirceREML

def verify_reml():
    np.random.seed(42)
    p = 3
    n = 150
    
    # Ground truth
    mu_true = np.array([10.0, -5.0, 2.0])
    gamma_true = np.array([1, 1, 1]) # True variances
    
    # Generate data
    # z = z_nom + H(lambda - lambda_prior) + eps
    # Let lambda_prior = 0 for simplicity of checking
    # z - z_nom = H * theta + eps, theta ~ N(mu_true, diag(gamma_true)) assuming mean shift?
    # Actually parameter is latent variable lambda. 
    # Current code assumes theta ~ N(mean, cov).
    # z_exp = z_nom + H * theta + eps
    
    H = np.random.randn(p, n)
    theta = np.random.normal(mu_true, np.sqrt(gamma_true)) # Usually theta is fixed for one realization? 
    # No, theta is the parameter vector we are estimating distribution of?
    # In this problem setting, we have n observations. 
    # Usually in regression: y_i = x_i^T beta + eps_i.
    # Here z_exp is vector of n observations.
    # The model is z = z_nom + H * random_param + eps ? 
    # If H is (p, n), then H * theta is (n,). theta is (p,).
    # Wait, H is usually (n, p) in stats, but here p, n = h.shape -> H is (p, n).
    # so H^T * theta ? Let's check code.
    # resid = z_exp - z_nom - np.sum(h * mean[:, None], axis=0)
    # mean is (p,), h is (p, n). broadcast -> (p, n). sum axis 0 -> (n,).
    # So yes, predicted = H^T * mean.
    
    # Let's generate data consistent with this.
    
    # True mean and covariance for generating single realization of data?
    # Or is it a random effects model where we see groups?
    # Circe seems to be: SINGLE group of data. We want to estimate mean and covariance of the parameters 
    # that best explain the discrepancy z_exp - z_nom.
    # But with only one realization of z_exp, we can't estimate Variance of parameters unless we have priors or structure?
    # Ah, the variance of the DATA is V = H^T Gamma H + Sigma_eps.
    # We estimate Gamma from the residuals.
    
    theta_sample = mu_true # Just one true parameter vector effectively?
    # If theta is fixed unknown, it's fixed effects.
    # If theta ~ N(mu, Gamma), and we observe z = H^t theta + eps.
    # Then z ~ N(H^t mu, H^t Gamma H + Sigma).
    # We want to find mu, Gamma.
    
    # Generate z
    # V_data = diag(sig_eps^2) + H^T diag(gamma) H  <-- Wait, existing code uses diagonal APPROX of this.
    # denom = sig_eps**2 + np.sum(gamma[:, None] * h**2, axis=0) -> diag(V_data) approx.
    
    # Let's generate data using the diagonal approximation assumption to be fair to the model.
    sig_eps = 0.01 * np.ones(n)
    
    # We need to simulate z such that its variance is close to what simple diagonal model expects?
    # Or just simulate real multivariate normal.
    
    # Covariance of z
    cov_theta = np.diag(gamma_true)
    cov_z = H.T @ cov_theta @ H + np.diag(sig_eps**2)
    
    z_nom = np.zeros(n)
    z_exp = np.random.multivariate_normal(H.T @ mu_true, cov_z)
    
    reml = CirceREML(
        initial_mean=np.zeros((p, 1)),
        initial_cov=np.eye(p),
        h=H,
        z_exp=z_exp,
        z_nom=z_nom,
        sig_eps=sig_eps,
        niter=2000
    )
    
    mean_list, cov_list, loglik_list, _, _ = reml.estimate(n_starts=50)
    
    est_mean = mean_list[-1].flatten()
    est_gamma = cov_list[-1].diagonal()
    
    print(f"True Mean: {mu_true}")
    print(f"Est Mean:  {est_mean}")
    print(f"True Gamma: {gamma_true}")
    print(f"Est Gamma:  {est_gamma}")
    
    # Check if NLL decreased
    print(f"Initial NLL: {-loglik_list[0]}")
    print(f"Final NLL:   {-loglik_list[-1]}")
    
    if -loglik_list[-1] < -loglik_list[0]:
        print("SUCCESS: NLL decreased.")
    else:
        print("FAILURE: NLL did not decrease.")
        
    return 0

if __name__ == "__main__":
    verify_reml()
