"""
Utilities for analyzing read count data stored as pandas DataFrames.
Includes library size normalization, mean-variance relationship analysis,
functions for differential expression testing.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.multitest import multipletests

from typing import Tuple

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
