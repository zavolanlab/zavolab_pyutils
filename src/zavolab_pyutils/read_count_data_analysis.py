"""
Utilities for analyzing read count data stored as pandas DataFrames.
Includes library size normalization, mean-variance relationship analysis,
funtions for differential expression testing and differetial relative usage testing 
(e.g. alternative splicing, AS and alternative polyadenylation, APA).
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import multiprocessing as mp

from scipy import stats
from scipy.interpolate import interp1d

from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

from scipy.optimize import minimize, OptimizeWarning
import warnings

from statsmodels.stats.multitest import multipletests

from typing import Tuple
from scipy.special import digamma, polygamma

def apply_deseq2_normalization(
    counts_df: pd.DataFrame, 
    metadata_df: pd.DataFrame, 
    sample_col: str = 'sample', 
    cond_col: str = 'condition', 
    lowExprGenesQ: float = 0.3, 
    pseudocount: float = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs DESeq2-style median-of-ratios normalization.
    
    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples). May contain additional annotation columns.
    metadata_df : pd.DataFrame
        Metadata mapping samples to biological conditions.
    sample_col : str, optional
        Column name in metadata_df containing sample IDs. Default is 'sample'.
    cond_col : str, optional
        Column name in metadata_df containing condition labels. Default is 'condition'.
    lowExprGenesQ : float, optional
        Quantile specifying the threshold to discard low-expressed genes 
        for size factor calculation. Default is 0.3.
    pseudocount : float, optional
        Added count before dividing by size factor value. Essential if further 
        log transformation is performed. Default is 1.
        
    Returns
    -------
    norm_counts_df : pd.DataFrame
        Dataframe of normalized counts (same shape as samples in metadata_df).
    sfs_df : pd.DataFrame
        Dataframe containing calculated size factors and read sums.
    """
    # 1. VALIDATION
    if sample_col not in metadata_df.columns:
        raise ValueError(f"Column '{sample_col}' not found in metadata_df.")
        
    if metadata_df[sample_col].duplicated().any():
        duplicated_samples = metadata_df[metadata_df[sample_col].duplicated()][sample_col].tolist()
        raise ValueError(f"Duplicate entries found in metadata column '{sample_col}': {duplicated_samples}.")
        
    sample_list = metadata_df[sample_col].tolist()
    missing_samples = [s for s in sample_list if s not in counts_df.columns]
    
    if missing_samples:
        raise ValueError(f"Missing samples in counts_df: {missing_samples}")

    # Work on a copy to avoid SettingWithCopy warnings on the original df
    df_work = counts_df[sample_list].copy()

    # 2. Filter low-expressed genes from Size Factor calculation
    # We use log-space to prevent overflow: geometric_mean = 2**(mean(log(x)))
    # We only care about genes with >0 counts in ALL samples for the reference test
    df_work['mean'] = df_work[sample_list].mean(axis=1)
    threshold = max(df_work['mean'].quantile(lowExprGenesQ), 0)
    
    high_expr_genes = df_work[(df_work['mean'] > threshold) & (df_work[sample_list].min(axis=1) > 0)].index
    
    # Slice only highly expressed genes
    ref_genes_df = df_work.loc[high_expr_genes].copy()
    
    # 3. Calculate Log Geometric Mean
    # No pseudocount needed here since we already filtered out zero-count genes
    log_counts = np.log2(ref_genes_df[sample_list]) 
    log_geom_means = log_counts.mean(axis=1)
    
    # 4. Calculate Size Factors (Median of Ratios)
    ratios = log_counts.sub(log_geom_means, axis=0)
    log2_sf_series = ratios.median(axis=0)

    # 5. Create the Size Factor DataFrame
    sfs_df = pd.DataFrame({
        'sample': log2_sf_series.index,
        'log2_sf': log2_sf_series.values
    })
    sfs_df.index = sfs_df['sample']
    sfs_df['sf'] = 2**(sfs_df['log2_sf'])
    
    # 6. Calculate Total Read Sums (for metadata/QC)
    # Fixed: Only sum the actual sample columns, ignoring the 'mean' column
    read_sums = df_work[sample_list].sum(axis=0)
    sfs_df['read_sum'] = sfs_df['sample'].map(read_sums)
    sfs_df['read_sum_mln'] = np.round(sfs_df['read_sum'] / 1e6, 2)
    
    # 7. Apply Normalization to the original matrix
    norm_counts_df = (df_work[sample_list] + pseudocount).div(sfs_df['sf'], axis=1)
    
    return norm_counts_df, sfs_df

def get_MultiDimR2(x, groups, R2adjusted=True):
    """
    Calculates the PERMANOVA R2 value for a given dataset and sample grouping.
    
    Parameters
    ----------
    x : np.ndarray
        Data matrix (samples (rows) x features (columns)) for which to calculate PERMANOVA R2 along samples.
    groups : list or np.ndarray
        A 1D sequence (list, array, or pandas Series) of group labels for each sample.
        Must have the exact same length as the number of rows in `x`.
    R2adjusted : bool, optional
        Whether to calculate the adjusted R2 value. Default is True.
        
    Returns
    -------
    R2 : float
        The PERMANOVA R2 value for the given dataset and sample grouping.
    """
    # Ensure inputs are numpy arrays for efficient masking
    x = np.asarray(x)
    groups = np.asarray(groups)
    
    # 1. Validation Check: Match rows in x to the length of groups
    if x.shape[0] != groups.shape[0]:
        raise ValueError(
            f"Shape mismatch: The data matrix 'x' has {x.shape[0]} samples (rows), "
            f"but {groups.shape[0]} group labels were provided."
        )
    
    # Calculate Total Sum of Squares (TSS)
    centroid = np.mean(x, axis=0)
    all_distances = pairwise_distances(x, [centroid], metric='sqeuclidean')
    TSS = np.sum(all_distances)
    
    # Calculate Residual Sum of Squares (RSS)
    RSS = 0
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        # Create a boolean mask for the current group
        mask = (groups == group)
        group_data = x[mask]
        
        # Calculate distances to the group-specific centroid
        if len(group_data) > 0:
            group_centroid = np.mean(group_data, axis=0)
            group_distances = pairwise_distances(group_data, [group_centroid], metric='sqeuclidean')
            RSS += np.sum(group_distances)
            
    # Calculate R2
    if R2adjusted:
        df_RSS = len(x) - len(unique_groups) - 1
        df_total = len(x) - 1
        R2 = 1 - (RSS / df_RSS) / (TSS / df_total)
    else:
        R2 = 1 - RSS / TSS
    
    return R2

def model_mean_variance(
    norm_counts_df, 
    metadata_df, 
    sample_col='sample', 
    cond_col='condition',
    CI_limit=0.95,
    outlier_q=0.9,
    max_iter_QuantReg=1000):
    """
    Estimates the mean-variance relationship using Quantile Regression.
    Useful for Negative Binomial / DESeq2 normalized counts.
    
    Parameters
    ----------
    norm_counts_df : pd.DataFrame
        Normalized count matrix.
    metadata_df : pd.DataFrame
        Metadata mapping samples to biological conditions.
    sample_col : str, optional
        Sample column name. Default is 'sample'.
    cond_col : str, optional
        Condition column name. Default is 'condition'.
    outlier_q : float, optional
        Quantile used to filter extreme outliers before model fitting. Default is 0.9.
        
    Returns
    -------
    RegrModel_df : pd.DataFrame
        DataFrame containing the fitted dispersion (alpha) parameter per condition.
    all_plot_data : pd.DataFrame
        DataFrame containing the mean, variance, and predicted variance for diagnostics.
    """
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = norm_counts_df.columns.intersection(sample_map.index)
    
    resid_df = norm_counts_df[common_samples].copy()
    sample_map = sample_map[common_samples]
    
    regr_models = []
    plot_data_list = []
    
    # Correction for Multiple testing across conditions using Bonferroni approach
    N_conditions = len(sample_map.unique())
    adjCI_limit = 1 - (1 - CI_limit) / N_conditions
    Low_q = (1 - adjCI_limit) / 2
    Up_q = 1 - Low_q
    
    for cond in sample_map.unique():
        samples_in_cond = sample_map[sample_map == cond].index
        n_reps = len(samples_in_cond)
        
        if n_reps < 2:
            print(f"Skipping {cond}: Need at least 2 replicates to compute variance.")
            continue 
        
        cond_data = resid_df[samples_in_cond].values
        
        # Calculate exact empirical statistics per gene for this condition
        gene_means = np.mean(cond_data, axis=1)
        gene_vars = np.var(cond_data, axis=1, ddof=1)
        
        df_cond = pd.DataFrame({
            'mean': gene_means,
            'var': gene_vars,
            'mean_2': gene_means**2,
            'condition': cond
        }, index=resid_df.index)
        
        # Exclude zeros and extreme outliers to ensure a robust fit
        data = df_cond[(df_cond['mean'] > 0) & (df_cond['mean'] < df_cond['mean'].quantile(outlier_q))].copy()
        
        # Model the overdispersion using Quantile Regression
        X = data[['mean_2']].values
        y = (data['var'] - data['mean']).values # Fit: var - mean
        
        mod = QuantReg(endog=y, exog=X)
        reg = mod.fit(q=0.5, max_iter=max_iter_QuantReg)
        
        data['QuantReGpred_var'] = reg.predict(X) + data['mean']
        
        regr_models.append([cond, 'QuantReg', 'var', reg.params[0]])
        plot_data_list.append(data)
        
    RegrModel_df = pd.DataFrame(regr_models, columns=['condition', 'model_type', 'pred_feature', 'param'])
    all_plot_data = pd.concat(plot_data_list)
    
    return RegrModel_df, all_plot_data

def get_deseq2_means_and_errors(norm_counts_df, metadata_df, regr_model_df, sample_col='sample', cond_col='condition'):
    """
    Calculates the mean expression and standard error for each condition 
    based on DESeq2 normalized counts and the fitted Quantile Regression dispersion model.
    
    Parameters
    ----------
    norm_counts_df : pd.DataFrame
        DESeq2 normalized count matrix.
    metadata_df : pd.DataFrame
        Metadata mapping samples to biological conditions.
    regr_model_df : pd.DataFrame
        The output from `model_mean_variance` containing the fitted alpha per condition.
    sample_col : str, optional
        Sample column name. Default is 'sample'.
    cond_col : str, optional
        Condition column name. Default is 'condition'.
        
    Returns
    -------
    means_df : pd.DataFrame
        Mean linear expression level per condition.
    errors_df : pd.DataFrame
        Standard Error of the Mean (SEM) per condition, derived from the Negative Binomial variance.
    """
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = norm_counts_df.columns.intersection(sample_map.index)
    
    df_work = norm_counts_df[common_samples].copy()
    sample_map = sample_map[common_samples]
    conditions = sample_map.unique()
    
    means_dict = {}
    errors_dict = {}
    
    for cond in conditions:
        samples_in_cond = sample_map[sample_map == cond].index
        n_reps = len(samples_in_cond)
        
        # 1. Calculate the raw mean
        cond_data = df_work[samples_in_cond].values
        mu = np.mean(cond_data, axis=1)
        means_dict[cond] = mu
        
        # 2. Retrieve the condition-specific dispersion (alpha)
        alpha_row = regr_model_df[regr_model_df['condition'] == cond]
        if not alpha_row.empty:
            alpha = alpha_row['param'].values[0]
        else:
            print(f"Warning: No fitted alpha found for {cond}. Using empirical variance.")
            alpha = None
            
        # 3. Calculate Variance and SEM
        if alpha is not None:
            # Negative Binomial modeled variance: V = mu + alpha * mu^2
            var = mu + alpha * (mu ** 2)
            var = np.maximum(var, 0) # Safety catch
        else:
            var = np.var(cond_data, axis=1, ddof=1)
            
        sem = np.sqrt(var / n_reps)
        errors_dict[cond] = sem
        
    means_df = pd.DataFrame(means_dict, index=df_work.index)
    errors_df = pd.DataFrame(errors_dict, index=df_work.index)
    
    return means_df, errors_df



#####
# Sanity Bayesian Normalization Implementation adapted from PMID: 33927416
#####

# --- TOP-LEVEL MULTIPROCESSING WORKERS ---
def _sanity_pass1_worker(args):
    i, n_g, exp_n_g, n_samples, min_variance = args
    def get_deltas(v):
        d = np.zeros(n_samples)
        for j in range(n_samples):
            current_d = np.log(max(n_g[j], min_variance) / exp_n_g[j]) if n_g[j] > 0 else 0.0
            for _ in range(30):
                exp_d = np.exp(current_d)
                f = n_g[j] - exp_n_g[j] * exp_d - current_d / v
                df = -exp_n_g[j] * exp_d - 1.0 / v
                step = f / df
                current_d -= step
                if abs(step) < min_variance: break
            d[j] = current_d
        return d

    def neg_log_evidence(v_scalar):
        v = v_scalar[0]
        d = get_deltas(v)
        term1 = (d ** 2) / (2 * v)
        term2 = -n_g * d
        term3 = exp_n_g * np.exp(d)
        term4 = 0.5 * np.log(1 + v * exp_n_g * np.exp(d))
        return np.sum(term1 + term2 + term3 + term4)

    with warnings.catch_warnings():
        # Specifically target the math warnings generated during inference bounds
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=OptimizeWarning)        
        # Use dynamic lower bound
        res = minimize(neg_log_evidence, x0=[0.1], bounds=[(min_variance, 20.0)])
    return i, res.x[0]

def _sanity_pass3_worker(args):
    i, n_g, exp_n_g, n_samples, v_optimal, min_variance = args
    
    # Safety catch: ensure v_optimal never mathematically hits pure zero due to float rounding
    v_optimal = max(v_optimal, min_variance)
    
    d = np.zeros(n_samples)
    for j in range(n_samples):
        current_d = np.log(max(n_g[j], min_variance) / exp_n_g[j]) if n_g[j] > 0 else 0.0
        for _ in range(30):
            exp_d = np.exp(current_d)
            f = n_g[j] - exp_n_g[j] * exp_d - current_d / v_optimal
            df = -exp_n_g[j] * exp_d - 1.0 / v_optimal
            step = f / df
            current_d -= step
            if abs(step) < min_variance: break
        d[j] = current_d
        
    opt_vars = 1.0 / (exp_n_g * np.exp(d) + 1.0 / v_optimal)
    return i, d, opt_vars

def apply_sanity_normalization(counts_df, metadata_df, sample_col='sample', cond_col='condition', empirical_bayes=False, n_cores=None, min_variance=1e-12):
    """
    Applies parallelized Sanity Bayesian normalization to RNA-seq counts.
    """
    if metadata_df[sample_col].duplicated().any():
        raise ValueError(f"Duplicate entries found in metadata column '{sample_col}'.")
    
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = counts_df.columns.intersection(sample_map.index)
    counts = counts_df[common_samples].values
    sample_map = sample_map[common_samples]
    conditions = sample_map.unique()
    
    gene_sums = counts.sum(axis=1)
    valid_mask = gene_sums > 0
    counts = counts[valid_mask]
    genes = counts_df.index[valid_mask]
    n_genes, n_samples = counts.shape
    
    N_c = counts.sum(axis=0)
    total_counts = N_c.sum()
    alpha_g = counts.sum(axis=1) / total_counts 
    expected_n = np.outer(alpha_g, N_c)
    
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
        
    print(f"PASS 1: Running Sanity inference on {n_genes} genes using {n_cores} cores...")
    pass1_args = [(i, counts[i, :], expected_n[i, :], n_samples, min_variance) for i in range(n_genes)]
    
    with mp.Pool(processes=n_cores) as pool:
        pass1_results = pool.map(_sanity_pass1_worker, pass1_args)
        
    pass1_results.sort(key=lambda x: x[0])
    raw_v_g = np.array([x[1] for x in pass1_results])

    if empirical_bayes:
        print("PASS 1.5: Applying Empirical Bayes Variance Shrinkage (Overdispersion-Only Fit)...")
        df_trend = pd.DataFrame({'expr': np.log10(alpha_g), 'v_g': raw_v_g})
        df_trend = df_trend.sort_values('expr')
        
        # 1. Isolate ONLY the genes with clear biological overdispersion
        # This prevents the underdispersed "Poisson" genes from dragging the curve down
        mask_valid = df_trend['v_g'] > (min_variance * 100)
        valid_expr = df_trend.loc[mask_valid, 'expr'].values
        valid_vg = df_trend.loc[mask_valid, 'v_g'].values
        
        if len(valid_vg) > np.quantile(valid_vg, 0.1):
            # 2. Fit LOWESS only on the valid, overdispersed genes
            trend = sm.nonparametric.lowess(endog=valid_vg, exog=valid_expr, frac=0.2, return_sorted=False)
            
            # 3. Interpolate the trend line back to ALL genes (rescuing the zeros)
            interpolator = interp1d(valid_expr, trend, kind='linear', fill_value="extrapolate")
            global_trend = interpolator(df_trend['expr'].values)
        else:
            global_trend = np.full(n_genes, np.median(valid_vg) if len(valid_vg) > 0 else 0.01)
        
        # 4. Establish a safe biological floor
        min_floor = np.quantile(valid_vg, 0.05) if len(valid_vg) > 0 else 1e-3
        df_trend['trended_v_g'] = np.clip(global_trend, a_min=min_floor, a_max=None)
        
        # 5. Chi-Squared small-sample bias correction
        df = max(1, n_samples - 1)
        chi2_median = stats.chi2.ppf(0.5, df)
        correction_factor = n_samples / chi2_median
        
        df_trend['robust_v_g'] = df_trend['trended_v_g'] * correction_factor
        
        # DESeq2 Strategy: Push underdispersed genes UP to the robust trend.
        # Leave genes that are highly overdispersed alone (or pull them toward the trend).
        # For Sanity, strictly using the smoothed trend for all genes ensures stable, 
        # non-zero CVs and prevents extreme false positives in DE testing.
        df_trend = df_trend.sort_index()
        final_v_g = df_trend['robust_v_g'].values
    else:
        final_v_g = raw_v_g
        
    print("PASS 2: Finalizing Bayesian Posteriors...")
    pass3_args = [(i, counts[i, :], expected_n[i, :], n_samples, final_v_g[i], min_variance) for i in range(n_genes)]
    
    with mp.Pool(processes=n_cores) as pool:
        pass3_results = pool.map(_sanity_pass3_worker, pass3_args)
        
    pass3_results.sort(key=lambda x: x[0])
    deltas = np.array([x[1] for x in pass3_results])
    variances = np.array([x[2] for x in pass3_results])

    ln2 = np.log(2)
    deltas_log2 = deltas / ln2
    variances_log2 = variances / (ln2**2)
    
    median_lib_size = np.median(N_c)
    log2_base_expr = np.log2(alpha_g * median_lib_size + 1e-18)
    
    sample_log2_expr = log2_base_expr[:, np.newaxis] + deltas_log2
    sample_norm_counts_df = pd.DataFrame(sample_log2_expr, index=genes, columns=common_samples)
    
    cond_means = {}
    cond_errors = {}
    
    for cond in conditions:
        idx = np.where(sample_map == cond)[0]
        n_reps = len(idx)
        
        cond_means[cond] = log2_base_expr + np.mean(deltas_log2[:, idx], axis=1)
        empirical_var = np.var(deltas_log2[:, idx], axis=1, ddof=1) if n_reps > 1 else 0
        posterior_var = np.mean(variances_log2[:, idx], axis=1) 
        cond_errors[cond] = np.sqrt((empirical_var + posterior_var) / n_reps)

    means_df = pd.DataFrame(cond_means, index=genes)
    errors_df = pd.DataFrame(cond_errors, index=genes)
    vg_df = pd.DataFrame({'inferred_v_g': final_v_g, 'raw_v_g': raw_v_g}, index=genes)
    
    print("Sanity normalization complete.")
    return sample_norm_counts_df, means_df, errors_df, vg_df

def test_differential_expression(means_df, errors_df, cond_A, cond_B):
    """
    Performs a Bayesian Wald test for differential expression between two conditions 
    using estimated means and standard errors.
    
    Parameters
    ----------
    means_df : pd.DataFrame
        Estimated mean log2 expression level per condition.
    errors_df : pd.DataFrame
        Estimated standard errors per condition.
    cond_A : str
        Name of the primary condition (Numerator).
    cond_B : str
        Name of the reference condition (Denominator).
        
    Returns
    -------
    res_df : pd.DataFrame
        DataFrame containing log2FC (A - B), standard error, Z-score, p-value, and FDR.
    """
    log2fc = means_df[cond_A] - means_df[cond_B]
    se_diff = np.sqrt(errors_df[cond_A]**2 + errors_df[cond_B]**2)
    
    # Calculate Wald Z-scores and two-tailed p-values
    z_scores = log2fc / se_diff
    p_values = 2 * stats.norm.sf(np.abs(z_scores))
    
    # Calculate FDR (Benjamini-Hochberg)
    mask = ~np.isnan(p_values)
    padj = np.full_like(p_values, np.nan)
    if mask.sum() > 0:
        padj[mask] = multipletests(p_values[mask], method='fdr_bh')[1]
        
    res_df = pd.DataFrame({
        'log2FC': log2fc,
        'SE': se_diff,
        'Z_score': z_scores,
        'p_value': p_values,
        'padj': padj
    }, index=means_df.index)
    
    return res_df

####
# Note: The full Bayesian Sanity implementation is more computationally intensive than the point-estimate version.
# It integrates over a grid of variance values to compute marginalized posteriors
####

# --- FULL BAYESIAN MULTIPROCESSING WORKER ---
def _sanity_full_bayesian_worker(args):
    """
    Computes marginalized posteriors, including q_g integration for exact 
    digamma/trigamma global mean calculations.
    """
    # FIX: We now receive N_c and alpha_g_val separately instead of a pre-multiplied exp_n_g
    i, n_g, N_c, alpha_g_val, n_samples, v_grid, min_variance, eb_prior_mean, eb_prior_var = args
    
    # Calculate exp_n_g inside the worker for the Newton-Raphson solver
    exp_n_g = alpha_g_val * N_c
    
    numbin = len(v_grid)
    log_liks = np.zeros(numbin)
    deltas_grid = np.zeros((numbin, n_samples))
    variances_grid = np.zeros((numbin, n_samples))
    q_g_grid = np.zeros(numbin) 
    
    for k, v in enumerate(v_grid):
        d = np.zeros(n_samples)
        for j in range(n_samples):
            current_d = np.log(max(n_g[j], min_variance) / exp_n_g[j]) if n_g[j] > 0 else 0.0
            for _ in range(30):
                exp_d = np.exp(current_d)
                f = n_g[j] - exp_n_g[j] * exp_d - current_d / v
                df = -exp_n_g[j] * exp_d - 1.0 / v
                step = f / df
                current_d -= step
                if abs(step) < min_variance: break
            d[j] = current_d
            
        deltas_grid[k, :] = d
        
        term1 = (d ** 2) / (2 * v)
        term2 = -n_g * d
        term3 = exp_n_g * np.exp(d)
        term4 = 0.5 * np.log(1 + v * exp_n_g * np.exp(d))
        
        log_evidence = -np.sum(term1 + term2 + term3 + term4)
        
        if eb_prior_mean is not None and eb_prior_var is not None:
            log_prior = -0.5 * ((np.log(v) - eb_prior_mean)**2 / eb_prior_var)
            log_evidence += log_prior
            
        log_liks[k] = log_evidence
        variances_grid[k, :] = 1.0 / (exp_n_g * np.exp(d) + 1.0 / v)
        
        # FIX: Calculate q_g strictly using N_c, perfectly matching original Sanity C++
        q_g_grid[k] = np.log(np.sum(N_c * np.exp(d)))
        
    max_ll = np.max(log_liks)
    probs = np.exp(log_liks - max_ll)
    probs_sum = np.sum(probs)
    probs = probs / probs_sum if probs_sum > 0 else np.ones(numbin) / numbin
        
    expected_deltas = np.sum(probs[:, np.newaxis] * deltas_grid, axis=0)
    expected_vars = np.sum(probs[:, np.newaxis] * variances_grid, axis=0)
    var_of_expectations = np.sum(probs[:, np.newaxis] * (deltas_grid - expected_deltas)**2, axis=0)
    final_variances = expected_vars + var_of_expectations
    
    expected_v_g = np.sum(probs * v_grid)
    max_lik_v_g = v_grid[np.argmax(probs)]
    
    expected_q_g = np.sum(probs * q_g_grid)
    var_q_g = np.sum(probs * (q_g_grid - expected_q_g)**2)

    return i, expected_deltas, final_variances, expected_v_g, max_lik_v_g, expected_q_g, var_q_g


def apply_sanity_normalization_full_bayesian(
    counts_df: pd.DataFrame, 
    metadata_df: pd.DataFrame, 
    sample_col: str = 'sample', 
    cond_col: str = 'condition', 
    empirical_bayes: bool = False,
    shrinkage_strength: float = 1.0, 
    vmin: float = 0.001, 
    vmax: float = 50.0, 
    numbin: int = 160,
    min_variance: float = 1e-12,
    n_cores: int = None,
    loess_variance_threshold_q: float = 0.1,
):
    """
    Applies the strict full Bayesian Sanity normalization to RNA-seq counts.
    Optionally applies Empirical Bayes shrinkage by deriving a prior from population trends.
    """
    if metadata_df[sample_col].duplicated().any():
        raise ValueError(f"Duplicate entries found in metadata column '{sample_col}'.")
    
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = counts_df.columns.intersection(sample_map.index)
    counts = counts_df[common_samples].values
    sample_map = sample_map[common_samples]
    conditions = sample_map.unique()
    
    gene_sums = counts.sum(axis=1)
    valid_mask = gene_sums > 0
    counts = counts[valid_mask]
    genes = counts_df.index[valid_mask]
    n_genes, n_samples = counts.shape
    
    N_c = counts.sum(axis=0)
    total_counts = N_c.sum()
    alpha_g = counts.sum(axis=1) / total_counts 
    
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
        
    deltav = np.log(vmax / vmin) / (numbin - 1)
    v_grid = vmin * np.exp(deltav * np.arange(numbin))
    
    if empirical_bayes:
        print(f"PASS 1: Running Initial Full Bayesian inference on {n_genes} genes to estimate variances...")
        # FIX: Pass N_c array and alpha_g scalar to the worker
        worker_args = [(i, counts[i, :], N_c, alpha_g[i], n_samples, v_grid, min_variance, None, None) for i in range(n_genes)]
        with mp.Pool(processes=n_cores) as pool:
            pass1_results = pool.map(_sanity_full_bayesian_worker, worker_args)
            
        pass1_results.sort(key=lambda x: x[0])
        raw_v_g = np.array([x[4] for x in pass1_results]) 
        
        print("PASS 2: Applying Empirical Bayes Variance Shrinkage...")
        df_trend = pd.DataFrame({'expr': np.log10(alpha_g), 'v_g': raw_v_g})
        df_trend = df_trend.sort_values('expr')
        
        mask_valid = df_trend['v_g'] > df_trend['v_g'].quantile(loess_variance_threshold_q)
        valid_expr = df_trend.loc[mask_valid, 'expr'].values
        valid_vg = df_trend.loc[mask_valid, 'v_g'].values
        
        if len(valid_vg) > 10:
            trend = sm.nonparametric.lowess(endog=valid_vg, exog=valid_expr, frac=0.2, return_sorted=False)
            interpolator = interp1d(valid_expr, trend, kind='linear', fill_value="extrapolate")
            global_trend = interpolator(df_trend['expr'].values)
            log_resids = np.log(valid_vg) - np.log(np.clip(trend, 1e-10, None))
            prior_var = np.var(log_resids, ddof=1) if len(log_resids) > 1 else 1.0
        else:
            global_trend = np.full(n_genes, np.median(valid_vg) if len(valid_vg) > 0 else 0.01)
            prior_var = 1.0
            
        min_floor = np.quantile(valid_vg, 0.05) if len(valid_vg) > 0 else 1e-3
        df_trend['trended_v_g'] = np.clip(global_trend, a_min=min_floor, a_max=None)
        
        df = max(1, n_samples - 1)
        correction_factor = n_samples / stats.chi2.ppf(0.5, df)
        df_trend['robust_v_g'] = df_trend['trended_v_g'] * correction_factor
        df_trend = df_trend.sort_index()
        
        eb_prior_means = np.log(df_trend['robust_v_g'].values)
        eb_prior_var = max(prior_var / shrinkage_strength, 0.01)
        
        # FIX: Pass N_c array and alpha_g scalar to the worker
        worker_args = [(i, counts[i, :], N_c, alpha_g[i], n_samples, v_grid, min_variance, eb_prior_means[i], eb_prior_var) for i in range(n_genes)]
        with mp.Pool(processes=n_cores) as pool:
            results = pool.map(_sanity_full_bayesian_worker, worker_args)
    else:
        print(f"Running Full Bayesian inference on {n_genes} genes")
        # FIX: Pass N_c array and alpha_g scalar to the worker
        worker_args = [(i, counts[i, :], N_c, alpha_g[i], n_samples, v_grid, min_variance, None, None) for i in range(n_genes)]
        with mp.Pool(processes=n_cores) as pool:
            results = pool.map(_sanity_full_bayesian_worker, worker_args)
            
    results.sort(key=lambda x: x[0])
    deltas = np.array([x[1] for x in results])
    variances = np.array([x[2] for x in results])
    expected_v_g = np.array([x[3] for x in results])
    max_v_g = np.array([x[4] for x in results])
    expected_q_g = np.array([x[5] for x in results])
    var_q_g = np.array([x[6] for x in results])

    ln2 = np.log(2)
    deltas_log2 = deltas / ln2
    variances_log2 = variances / (ln2**2)
    
    # -------------------------------------------------------------
    # EXACT GLOBAL MEAN & TRIGAMMA ERROR BAR
    # -------------------------------------------------------------
    n_g_total = counts.sum(axis=1) 
    
    # Global mean LTQ is now accurately restored
    global_mean_nat = digamma(n_g_total + 1) - expected_q_g
    global_mean_log2 = global_mean_nat / ln2
    
    trigamma_term = polygamma(1, n_g_total + 1)
    global_error_var_nat = trigamma_term + var_q_g
    global_error_var_log2 = global_error_var_nat / (ln2**2)
    
    median_lib_size = np.median(N_c)
    sample_log2_expr = global_mean_log2[:, np.newaxis] + deltas_log2 + np.log2(median_lib_size)
    sample_norm_counts_df = pd.DataFrame(sample_log2_expr, index=genes, columns=common_samples)
    
    cond_means = {}
    cond_errors_relative = {} 
    cond_errors_absolute = {} 
    
    for cond in conditions:
        idx = np.where(sample_map == cond)[0]
        n_reps = len(idx)
        
        cond_means[cond] = global_mean_log2 + np.mean(deltas_log2[:, idx], axis=1) + np.log2(median_lib_size)
        
        empirical_var = np.var(deltas_log2[:, idx], axis=1, ddof=1) if n_reps > 1 else 0
        posterior_var = np.mean(variances_log2[:, idx], axis=1) 
        relative_var = (empirical_var + posterior_var) / n_reps
        
        cond_errors_relative[cond] = np.sqrt(relative_var)
        cond_errors_absolute[cond] = np.sqrt(relative_var + global_error_var_log2)

    means_df = pd.DataFrame(cond_means, index=genes)
    errors_relative_df = pd.DataFrame(cond_errors_relative, index=genes)
    errors_absolute_df = pd.DataFrame(cond_errors_absolute, index=genes)
    
    variances_df = pd.DataFrame(variances_log2, index=genes, columns=common_samples)
    
    vg_df = pd.DataFrame({
        'marginalized_v_g': expected_v_g, 
        'MAP_v_g': max_v_g,
        'global_mean_error_log2': np.sqrt(global_error_var_log2)
    }, index=genes)
    
    if empirical_bayes:
        vg_df.insert(0, 'raw_MAP_v_g', raw_v_g)
        vg_df.insert(1, 'trended_prior_v_g', df_trend['robust_v_g'].values)
    
    print("Full Bayesian Sanity normalization complete.")
    
    return sample_norm_counts_df, means_df, errors_relative_df, errors_absolute_df, vg_df, median_lib_size, variances_df

def test_differential_relative_usage(
    norm_counts_df: pd.DataFrame, 
    variances_df: pd.DataFrame, 
    metadata_df: pd.DataFrame, 
    isoform_pairs: list, 
    cond_A: str, 
    cond_B: str,
    sample_col: str = 'sample', 
    cond_col: str = 'condition'
) -> pd.DataFrame:
    """
    Tests for Differential Relative Usage (e.g., Alternative Splicing, APA) between two conditions
    using Full Bayesian cell-level posteriors.
    
    Parameters
    ----------
    norm_counts_df : pd.DataFrame
        normalized log2 expression estimates (output from Sanity).
    variances_df : pd.DataFrame
        log2 posterior variances (output from Sanity).
    metadata_df : pd.DataFrame
        Metadata mapping samples to conditions.
    isoform_pairs : list of tuples
        List of (Isoform_1_ID, Isoform_2_ID) to compare. (e.g., [('GeneX_Proximal', 'GeneX_Distal')])
    cond_A : str
        Primary condition (Numerator in the fold-change).
    cond_B : str
        Reference condition (Denominator in the fold-change).
        
    Returns
    -------
    res_df : pd.DataFrame
        DataFrame containing the Delta-Delta Log2 Ratio, Standard Error, Z-score, and FDR.
    """
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    
    cells_A = sample_map[sample_map == cond_A].index.intersection(norm_counts_df.columns)
    cells_B = sample_map[sample_map == cond_B].index.intersection(norm_counts_df.columns)
    
    n_A = len(cells_A)
    n_B = len(cells_B)
    
    results = []
    
    for iso1, iso2 in isoform_pairs:
        # Check if both isoforms survived filtering
        if iso1 not in norm_counts_df.index or iso2 not in norm_counts_df.index:
            continue
            
        # 1. Calculate Per-sample Log2 Ratios: log2(Iso1) - log2(Iso2)
        log2_ratio_cells = norm_counts_df.loc[iso1] - norm_counts_df.loc[iso2]
        
        # 2. Calculate Per-sample Posterior Variance of the Ratio
        # Because the Poisson draws are conditionally independent given the rates, 
        # the posterior uncertainty of the difference is the sum of the uncertainties.
        var_ratio_cells = variances_df.loc[iso1] + variances_df.loc[iso2]
        
        # 3. Aggregate for Condition A
        mean_ratio_A = log2_ratio_cells[cells_A].mean()
        # Empirical variance naturally captures the biological covariance Cov(Iso1, Iso2) across cells
        empirical_var_A = log2_ratio_cells[cells_A].var(ddof=1) if n_A > 1 else 0
        posterior_var_A = var_ratio_cells[cells_A].mean()
        se_ratio_A = np.sqrt((empirical_var_A + posterior_var_A) / n_A)
        
        # 4. Aggregate for Condition B
        mean_ratio_B = log2_ratio_cells[cells_B].mean()
        empirical_var_B = log2_ratio_cells[cells_B].var(ddof=1) if n_B > 1 else 0
        posterior_var_B = var_ratio_cells[cells_B].mean()
        se_ratio_B = np.sqrt((empirical_var_B + posterior_var_B) / n_B)
        
        # 5. Delta-Delta Log2 FC (Difference of the Ratios)
        dd_log2fc = mean_ratio_A - mean_ratio_B
        se_diff = np.sqrt(se_ratio_A**2 + se_ratio_B**2)
        
        z_score = dd_log2fc / se_diff if se_diff > 0 else 0
        p_val = 2 * stats.norm.sf(np.abs(z_score))
        
        results.append({
            'Pair': f"{iso1}_vs_{iso2}",
            'isoform_1': iso1,
            'isoform_2': iso2,
            'log2Ratio_A': mean_ratio_A,
            'log2Ratio_B': mean_ratio_B,
            'dd_log2FC': dd_log2fc,
            'SE': se_diff,
            'Z_score': z_score,
            'p_value': p_val
        })
        
    res_df = pd.DataFrame(results).set_index('Pair')
    
    # Calculate FDR (Benjamini-Hochberg)
    if not res_df.empty:
        mask = ~res_df['p_value'].isna()
        res_df.loc[mask, 'padj'] = multipletests(res_df.loc[mask, 'p_value'], method='fdr_bh')[1]
        
    return res_df

def prepare_isoform_sanity_matrix(iso_counts_df: pd.DataFrame, isoform_to_gene: dict) -> Tuple[pd.DataFrame, list]:
    """
    Creates an augmented count matrix containing both the original isoforms
    and a 'Rest_of_Gene' count for each isoform to maintain Poisson independence.
    
    Returns
    -------
    augmented_counts_df : pd.DataFrame
        Raw counts matrix ready for Sanity, containing Isoforms and Rest_of_Gene rows.
    iso_vs_rest_pairs : list of tuples
        List of pairs formatted for the test/plot functions.
    """
    # 1. Calculate Total Gene Counts
    gene_counts = pd.DataFrame(index=iso_counts_df.index)
    gene_mapping_df = pd.DataFrame(list(isoform_to_gene.items()), columns=['Isoform', 'Gene'])
    
    gene_sums = {}
    for gene, group in gene_mapping_df.groupby('Gene'):
        valid_isos = [iso for iso in group['Isoform'] if iso in iso_counts_df.index]
        if valid_isos:
            gene_sums[gene] = iso_counts_df.loc[valid_isos].sum(axis=0)
            
    gene_counts_df = pd.DataFrame(gene_sums).T
    
    # 2. Calculate "Rest of Gene" for each isoform
    rest_counts_dict = {}
    iso_vs_rest_pairs = []
    
    for iso, gene in isoform_to_gene.items():
        if iso in iso_counts_df.index and gene in gene_counts_df.index:
            # Rest of gene = Total Gene - This Isoform
            rest_counts_dict[f"{iso}_REST"] = gene_counts_df.loc[gene] - iso_counts_df.loc[iso]
            iso_vs_rest_pairs.append((iso, f"{iso}_REST"))
            
    rest_counts_df = pd.DataFrame(rest_counts_dict).T
    
    # 3. Combine into one augmented matrix
    augmented_counts_df = pd.concat([iso_counts_df, rest_counts_df])
    
    return augmented_counts_df, iso_vs_rest_pairs