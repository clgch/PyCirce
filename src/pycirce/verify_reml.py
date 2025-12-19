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
    p = 2
    n = 50
    
    # Ground truth
    mu_true = np.array([-1, 10])
    gamma_true = np.array([10, 1]) # True variances
    
    # Generate data
    # z = z_nom + H(lambda - lambda_prior) + eps
    # Let lambda_prior = 0 for simplicity of checking
    # z - z_nom = H * theta + eps, theta ~ N(mu_true, diag(gamma_true)) assuming mean shift?
    # Actually parameter is latent variable lambda. 
    # Current code assumes theta ~ N(mean, cov).
    # z_exp = z_nom + H * theta + eps
    
    H = np.random.randn(p, n)

    theta = np.random.multivariate_normal(mu_true, np.sqrt(np.diag(gamma_true)), size=n)
    sig_eps = 0.01 * np.ones(n)

    z_nom = np.zeros(n)
    z_exp = np.sum(H.T * theta, axis=1) + np.random.normal(0, sig_eps, size=n)
    
    # We need to simulate z such that its variance is close to what simple diagonal model expects?
    # Or just simulate real multivariate normal.
    
    # Covariance of z
    # cov_theta = np.diag(gamma_true)
    # cov_z = H.T @ cov_theta @ H + np.diag(sig_eps**2)
    
    # z_nom = np.zeros(n)
    # z_exp = np.random.multivariate_normal(H.T @ mu_true, cov_z)
    
    reml = CirceREML(
        initial_mean=np.zeros((p, 1)),
        initial_cov=np.eye(p),
        h=H,
        z_exp=z_exp,
        z_nom=z_nom,
        sig_eps=sig_eps,
        niter=15000
    )
    
    est_mean, est_gamma, loglik_list = reml.estimate(n_starts=50)
    
    
    print(f"True Mean: {mu_true}")
    print(f"Est Mean:  {est_mean}")
    print(f"True sqrt Gamma: {np.sqrt(gamma_true)}")
    print(f"Est sqrt Gamma:  {np.sqrt(est_gamma)}")
    
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
