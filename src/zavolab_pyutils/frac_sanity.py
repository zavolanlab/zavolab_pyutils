import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import norm

def fit_frac_sanity_params(ltq_total: np.ndarray, ltq_pd: np.ndarray):
    """
    Estimates the global pull-down fraction (F) and Log-Beta parameters (a, b).
    """
    # Calculate observed D_g
    D_obs = ltq_pd - ltq_total
    
    # Mathematical upper bound for F: F <= min(2^-D)
    # Using 1st percentile to protect against extreme outliers
    max_D = np.percentile(D_obs, 99) 
    max_F = min(1.0, 2.0 ** (-max_D))
    
    def negative_log_likelihood(params):
        F, a, b = params
        X = D_obs + np.log2(F)
        
        # Epsilon buffer to prevent log(0) and enforce strict X <= 0
        eps = 1e-10
        alpha = np.clip(2.0 ** X, a_min=eps, a_max=1.0 - eps)
        
        log_term_1 = a * X * np.log(2)
        log_term_2 = (b - 1.0) * np.log(1.0 - alpha)
        log_beta_norm = betaln(a, b)
        
        LL_genes = log_term_1 + log_term_2 - log_beta_norm
        return -np.sum(LL_genes)

    init_params = [min(0.1, max_F * 0.9), 1.0, 2.0]
    
    bounds = [
        (1e-5, max_F),    # F
        (0.01, 100.0),    # a
        (1.001, 100.0)    # b (prevents infinite spike at alpha=1)
    ]
    
    result = minimize(negative_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds)
    
    if not result.success:
        print(f"Warning: Optimization failed - {result.message}")
        
    return result.x[0], result.x[1], result.x[2] # F, a, b

def calculate_differential_recruitment(
    ltq_total_ut: np.ndarray, var_total_ut: np.ndarray,
    ltq_pd_ut: np.ndarray, var_pd_ut: np.ndarray,
    ltq_total_stress: np.ndarray, var_total_stress: np.ndarray,
    ltq_pd_stress: np.ndarray, var_pd_stress: np.ndarray,
    gene_ids: list
):
    """
    Computes true differential recruitment (Delta Log2 Alpha) and posterior significance.
    """
    # 1. Fit UT condition
    F_ut, a_ut, b_ut = fit_frac_sanity_params(ltq_total_ut, ltq_pd_ut)
    D_ut = ltq_pd_ut - ltq_total_ut
    log2_alpha_ut = D_ut + np.log2(F_ut)
    var_D_ut = var_pd_ut + var_total_ut
    
    # 2. Fit Stress condition
    F_stress, a_stress, b_stress = fit_frac_sanity_params(ltq_total_stress, ltq_pd_stress)
    D_stress = ltq_pd_stress - ltq_total_stress
    log2_alpha_stress = D_stress + np.log2(F_stress)
    var_D_stress = var_pd_stress + var_total_stress
    
    # 3. Calculate Differential Metrics
    delta_log2_alpha = log2_alpha_stress - log2_alpha_ut
    delta_log2_F = np.log2(F_stress) - np.log2(F_ut)
    posterior_variance = var_D_stress + var_D_ut
    
    # 4. Statistical significance (Z-test)
    z_scores = delta_log2_alpha / np.sqrt(posterior_variance)
    p_values = 2 * norm.sf(np.abs(z_scores)) # Two-tailed
    
    results_df = pd.DataFrame({
        'gene_id': gene_ids,
        'log2_alpha_UT': log2_alpha_ut,
        'log2_alpha_Stress': log2_alpha_stress,
        'delta_log2_alpha': delta_log2_alpha,
        'posterior_variance': posterior_variance,
        'z_score': z_scores,
        'p_value': p_values
    })
    
    metadata = {
        'F_UT': F_ut, 'F_Stress': F_stress, 'delta_log2_F': delta_log2_F,
        'Beta_params_UT': (a_ut, b_ut), 'Beta_params_Stress': (a_stress, b_stress)
    }
    
    return results_df, metadata