import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import norm, beta, kstest

def fit_frac_sanity_map(ltq_total: np.ndarray, ltq_pd: np.ndarray, prior_alpha_F: float = 1.0, prior_beta_F: float = 1.0):
    """
    MAP Estimation of Log-Beta parameters (a, b) with F analytically constrained.
    """
    D_obs = ltq_pd - ltq_total
    R_obs = 2.0 ** D_obs
    mean_ratio = np.mean(R_obs)
    
    # Cap expected value to satisfy alpha_g <= 1.0
    max_D = np.percentile(D_obs, 99.9) 
    max_F_bound = min(1.0 - 1e-7, 2.0 ** (-max_D))
    
    is_uniform_prior = (prior_alpha_F == 1.0 and prior_beta_F == 1.0)
    
    def negative_log_posterior(params):
        a, b = params
        F = a / ((a + b) * mean_ratio)
        
        if F <= 0 or F > max_F_bound:
            return np.inf
            
        X = D_obs + np.log2(F)
        eps = 1e-10
        alpha = np.clip(2.0 ** X, a_min=eps, a_max=1.0 - eps)
        
        # 1. Negative Log-Likelihood[cite: 5]
        log_term_1 = a * X * np.log(2)
        log_term_2 = (b - 1.0) * np.log(1.0 - alpha)
        log_beta_norm = betaln(a, b)
        NLL = -np.sum(log_term_1 + log_term_2 - log_beta_norm)
        
        # 2. Negative Log-Prior on F[cite: 5]
        if is_uniform_prior:
            NLP = NLL
        else:
            log_prior_F = (prior_alpha_F - 1) * np.log(F) + (prior_beta_F - 1) * np.log(1 - F) - betaln(prior_alpha_F, prior_beta_F)
            NLP = NLL - log_prior_F
            
        return NLP

    init_params = [1.0, 19.0] 
    bounds = [(0.01, 1000.0), (1.001, 1000.0)] # a > 0, b >= 1.001[cite: 5]
    
    result = minimize(negative_log_posterior, init_params, method='L-BFGS-B', bounds=bounds)
    
    a_map, b_map = result.x
    F_map = a_map / ((a_map + b_map) * mean_ratio)
    
    # Laplace Approximation for Credible Intervals via Inverse Hessian[cite: 5]
    try:
        cov_ab = result.hess_inv.todense()
        var_a = cov_ab[0, 0]
        var_b = cov_ab[1, 1]
        
        # Delta method to approximate Var(F)[cite: 5]
        df_da = b_map / (((a_map + b_map)**2) * mean_ratio)
        df_db = -a_map / (((a_map + b_map)**2) * mean_ratio)
        grad_F = np.array([df_da, df_db])
        
        var_F = grad_F.T @ cov_ab @ grad_F
        var_log2_F = var_F / ((F_map * np.log(2))**2) # Delta method for log-space[cite: 5]
    except Exception:
        var_F = np.nan
        var_log2_F = np.nan

    # Calculate specific recruitments to test goodness-of-fit[cite: 5]
    # alpha_g is a physical probability and must stay within [0, 1] (see PDF Sec. 1 & 3.4)
    alpha_g = np.clip(F_map * R_obs, 0.0, 1.0)
    ks_stat, ks_pval = kstest(alpha_g, 'beta', args=(a_map, b_map))
        
    return {
        'F': F_map, 'var_F': var_F, 'var_log2_F': var_log2_F,
        'a': a_map, 'b': b_map,
        'ks_stat': ks_stat, 'ks_pval': ks_pval,
        'alpha_g': alpha_g,
        'success': result.success,
        'is_mle': is_uniform_prior
    }

def calculate_differential_recruitment(
    ltq_total_ut, var_total_ut, ltq_pd_ut, var_pd_ut,
    ltq_total_stress, var_total_stress, ltq_pd_stress, var_pd_stress,
    gene_ids, prior_ut=(1.0, 1.0), prior_stress=(1.0, 1.0)
):
    """
    Computes differential recruitment, MAP estimates, and Bayesian credible intervals[cite: 5].
    """
    fit_ut = fit_frac_sanity_map(ltq_total_ut, ltq_pd_ut, prior_ut[0], prior_ut[1])
    fit_stress = fit_frac_sanity_map(ltq_total_stress, ltq_pd_stress, prior_stress[0], prior_stress[1])
    
    D_ut = ltq_pd_ut - ltq_total_ut
    D_stress = ltq_pd_stress - ltq_total_stress
    
    log2_alpha_ut = D_ut + np.log2(fit_ut['F'])
    log2_alpha_stress = D_stress + np.log2(fit_stress['F'])
    
    delta_log2_alpha = log2_alpha_stress - log2_alpha_ut
    delta_log2_F = np.log2(fit_stress['F']) - np.log2(fit_ut['F'])
    var_delta_log2_F = fit_stress['var_log2_F'] + fit_ut['var_log2_F'] # Sum of log-variances[cite: 5]
    
    # Posterior Variance from pySanity LTQs[cite: 5]
    posterior_variance = var_pd_stress + var_total_stress + var_pd_ut + var_total_ut
    
    z_scores = delta_log2_alpha / np.sqrt(posterior_variance)
    p_values = 2 * norm.sf(np.abs(z_scores))
    
    results_df = pd.DataFrame({
        'gene_id': gene_ids,
        'log2_alpha_UT': log2_alpha_ut,
        'log2_alpha_Stress': log2_alpha_stress,
        'delta_log2_alpha': delta_log2_alpha,
        'posterior_variance': posterior_variance,
        'z_score': z_scores,
        'p_value': p_values,
        # Per-condition V_g = Var(LTQ_PD,g) + Var(LTQ_Total,g), used for single-condition CIs
        'var_D_UT': var_pd_ut + var_total_ut,
        'var_D_Stress': var_pd_stress + var_total_stress,
        # Credible intervals for natural sub-fractions[cite: 5]
        'alpha_UT': fit_ut['alpha_g'],
        'alpha_UT_CI_lower': 2.0**(log2_alpha_ut - 1.96 * np.sqrt(var_pd_ut + var_total_ut)),
        'alpha_UT_CI_upper': 2.0**(log2_alpha_ut + 1.96 * np.sqrt(var_pd_ut + var_total_ut)),
        'alpha_Stress': fit_stress['alpha_g'],
        'alpha_Stress_CI_lower': 2.0**(log2_alpha_stress - 1.96 * np.sqrt(var_pd_stress + var_total_stress)),
        'alpha_Stress_CI_upper': 2.0**(log2_alpha_stress + 1.96 * np.sqrt(var_pd_stress + var_total_stress)),
    })
    
    metadata = {
        'fit_UT': fit_ut, 
        'fit_Stress': fit_stress, 
        'F_UT': fit_ut['F'],
        'F_Stress': fit_stress['F'],
        'delta_log2_F': delta_log2_F,
        'var_delta_log2_F': var_delta_log2_F
    }
    return results_df, metadata