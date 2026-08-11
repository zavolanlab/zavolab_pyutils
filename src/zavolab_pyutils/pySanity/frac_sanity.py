import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, roots_hermite
from scipy.stats import norm, kstest

def fit_frac_sanity_map(ltq_total: np.ndarray, var_total: np.ndarray, ltq_pd: np.ndarray, var_pd: np.ndarray, prior_alpha_F: float = 1.0, prior_beta_F: float = 1.0):
    """
    MAP Estimation of Log-Beta parameters (a, b) with F analytically constrained.
    Implements the exact likelihood formula convolved with Gaussian measurement error,
    stabilized via the Log-Sum-Exp trick for accurate Hessian approximation.
    """
    ln2 = np.log(2)
    D_obs = ltq_pd - ltq_total
    V_obs = np.clip(var_pd + var_total, 1e-12, None)
    
    # 1. EXACT PDF FORMULATION: Arithmetic mean of the observed 2^D_g ratios
    R_obs = 2.0 ** D_obs
    mean_ratio = np.mean(R_obs)
    
    is_uniform_prior = (prior_alpha_F == 1.0 and prior_beta_F == 1.0)
    
    # 2. Setup Gauss-Hermite Quadrature
    n_quad = 15
    z_k, w_k = roots_hermite(n_quad)
    log_w_k = np.log(w_k)
    
    D_true_grid = D_obs[:, np.newaxis] + np.sqrt(2.0 * V_obs)[:, np.newaxis] * z_k[np.newaxis, :]
    
    def negative_log_posterior(params):
        a, b = params
        
        # Analytic calculation of F[cite: 7]
        F = a / ((a + b) * mean_ratio)
        
        if F <= 1e-7 or F >= 1.0 - 1e-7:
            return 1e9 + 1e5 * (F - 0.5)**2
            
        alpha_grid = F * (2.0 ** D_true_grid)
        
        # Mask valid physical probabilities
        valid_mask = (alpha_grid > 1e-10) & (alpha_grid < 1.0 - 1e-10)
        
        # Initialize log-likelihood array for Log-Sum-Exp
        L = np.full_like(alpha_grid, -np.inf)
        
        if np.any(valid_mask):
            alpha_safe = alpha_grid[valid_mask]
            # P(D_g | a, b) evaluated purely in log-space[cite: 7]
            L[valid_mask] = (
                a * np.log(alpha_safe) + 
                (b - 1.0) * np.log(1.0 - alpha_safe) - 
                betaln(a, b) + 
                np.log(ln2)
            ) + log_w_k[np.newaxis, :].repeat(D_obs.shape[0], axis=0)[valid_mask]
        
        # 3. LOG-SUM-EXP TRICK: log(sum(exp(L))) = L_max + log(sum(exp(L - L_max)))
        L_max = np.max(L, axis=1, keepdims=True)
        L_max_finite = np.where(np.isinf(L_max), 0, L_max) 
        
        sum_exp = np.sum(np.exp(L - L_max_finite), axis=1)
        
        invalid_genes = np.isinf(L_max).flatten() | (sum_exp <= 0)
        
        log_marginal = np.zeros(D_obs.shape[0])
        log_marginal[~invalid_genes] = -0.5 * np.log(np.pi) + L_max_finite[~invalid_genes].flatten() + np.log(sum_exp[~invalid_genes])
        log_marginal[invalid_genes] = -1e5  # Steep penalty for physically impossible regions
        
        NLL = -np.sum(log_marginal)
        
        if is_uniform_prior:
            NLP = NLL
        else:
            log_prior_F = (prior_alpha_F - 1) * np.log(F) + (prior_beta_F - 1) * np.log(1 - F) - betaln(prior_alpha_F, prior_beta_F)
            NLP = NLL - log_prior_F
            
        return NLP

    # 4. Multi-start initialization to prevent local-minima trapping
    best_nlp = np.inf
    best_res = None
    bounds = [(0.01, 1000.0), (1.001, 1000.0)]
    
    initializations = [(1.0, 19.0), (2.0, 8.0), (5.0, 5.0), (0.5, 50.0)]
    
    for init_a, init_b in initializations:
        try:
            # Skip initializations that violate the analytical F boundary
            test_F = init_a / ((init_a + init_b) * mean_ratio)
            if test_F >= 1.0: 
                continue
                
            res = minimize(negative_log_posterior, [init_a, init_b], method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nlp:
                best_nlp = res.fun
                best_res = res
        except Exception:
            pass
            
    if best_res is None:
        a_map, b_map = 1.0, 19.0
        opt_success = False
    else:
        a_map, b_map = best_res.x
        opt_success = best_res.success
        
    F_map = a_map / ((a_map + b_map) * mean_ratio)
    
    # 5. Laplace Approximation for Credible Intervals via Inverse Hessian[cite: 7]
    try:
        cov_ab = best_res.hess_inv.todense()
        var_a = cov_ab[0, 0]
        var_b = cov_ab[1, 1]
        
        # Delta method for Var(F)[cite: 7]
        df_da = b_map / (((a_map + b_map)**2) * mean_ratio)
        df_db = -a_map / (((a_map + b_map)**2) * mean_ratio)
        grad_F = np.array([df_da, df_db])
        
        var_F = grad_F.T @ cov_ab @ grad_F
        var_log2_F = var_F / ((F_map * ln2)**2) 
    except Exception:
        var_F = np.nan
        var_log2_F = np.nan

    alpha_g = np.clip(F_map * (2.0 ** D_obs), 0.0, 1.0)
    ks_stat, ks_pval = kstest(alpha_g, 'beta', args=(a_map, b_map))
        
    return {
        'F': F_map, 'var_F': var_F, 'var_log2_F': var_log2_F,
        'a': a_map, 'b': b_map,
        'ks_stat': ks_stat, 'ks_pval': ks_pval,
        'alpha_g': alpha_g,
        'success': opt_success,
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
    fit_ut = fit_frac_sanity_map(ltq_total_ut, var_total_ut, ltq_pd_ut, var_pd_ut, prior_ut[0], prior_ut[1])
    fit_stress = fit_frac_sanity_map(ltq_total_stress, var_total_stress, ltq_pd_stress, var_pd_stress, prior_stress[0], prior_stress[1])    

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