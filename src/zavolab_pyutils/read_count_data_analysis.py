"""
Utilities for analyzing read count data stored as pandas DataFrames.
Includes library size normalization, mean-variance relationship analysis,
funtions for differential expression testing and differetial relative usage testing 
(e.g. alternative splicing, AS and alternative polyadenylation, APA).
"""

import pandas as pd
import numpy as np

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
    high_expr_genes = df_work[(df_work['geom_mean'] > threshold)&(df_work[sample_list].min(axis=1) > 0)].index
    
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
    norm_counts = (df_work[sample_list]+pseudocount).div(sfs_df['sf'], axis=1)
    
    return norm_counts, sfs_df
