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

from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg

def deseq2_normalize(counts_df, sample_list, lowExprGenesQ=0.3, pseudocount = 1):
    """
    Performs DESeq2-style median-of-ratios normalization.
    
    Assumes inputs are validated (sample_list exists in counts_df columns).
    
    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples). May contain any additional columns, e.g. gene names, gene lengths etc.
    sample_list : list
        List of column names in counts_df to normalize.
    lowExprGenesQ: float
        the quantile specifying the threshold to discard low-expressed genes for size factor calculation
    pseudocount: float
        added count before dividing by size factor value. Essential if further log2 transformation is performed.
        
    Returns
    -------
    norm_counts : pd.DataFrame
        Dataframe of normalized counts (same shape as input).
    sfs_df : pd.DataFrame
        Dataframe containing calculated size factors and read sums.
    """

    # check that all samples in sample_list are present in counts_df
    missing_samples = [s for s in sample_list if s not in counts_df.columns]
    if missing_samples:
        raise ValueError(f"Missing samples in counts_df: {missing_samples}")

    # Work on a copy to avoid SettingWithCopy warnings on the original df
    # We only look at the specific samples requested
    df_work = counts_df[sample_list].copy()

    # 1. Filter low-expressed genes from Size Factor calculation
    # We use log-space to prevent overflow: geometric_mean = 2**(mean(log(x)))
    # We only care about genes with >0 counts in ALL samples for the reference
    # test
    df_work['mean'] = df_work.mean(axis=1)
    threshold = max(df_work['mean'].quantile(lowExprGenesQ),0)
    high_expr_genes = df_work[(df_work['mean'] > threshold)&(df_work[sample_list].min(axis=1) > 0)].index
    
    # Slice only highly expressed genes
    ref_genes_df = df_work.loc[high_expr_genes].copy()
    
    # 2. Calculate Log Geometric Mean
    log_counts = np.log2(ref_genes_df[sample_list]) # we don't need pseudocount here since we already filtered out zero-count genes
    log_geom_means = log_counts.mean(axis=1)
    
    # 3. Calculate Size Factors (Median of Ratios)
    # Ratio = Count / GeometricMean
    # We use the filtered high-expression set
    ratios = log_counts.sub(log_geom_means, axis=0)
    log2_sf_series = ratios.median(axis=0)

    # 4. Create the Size Factor DataFrame
    sfs_df = pd.DataFrame({
        'sample': log2_sf_series.index,
        'log2_sf': log2_sf_series.values
    })
    sfs_df.index = sfs_df['sample']
    sfs_df['sf'] = 2**(sfs_df['log2_sf'])
    
    # 5. Calculate Total Read Sums (for metadata/QC)
    read_sums = df_work.sum(axis=0)
    sfs_df['read_sum'] = read_sums
    sfs_df['read_sum_mln'] = np.round(sfs_df['read_sum'] / 1e6, 2)
    
    # 6. Apply Normalization to the original matrix
    # Apply pseudocount to original counts and divide by the size factors
    norm_counts_df = (df_work[sample_list]+pseudocount).div(sfs_df['sf'], axis=1)
    
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
    Estimates the mean-variance relationship within every condition individually
    to account for condition-specific between-replicate variability.
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