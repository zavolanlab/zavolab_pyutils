"""
Utilities for analyzing read count data stored as pandas DataFrames.
Includes library size normalization, mean-variance relationship analysis,
funtions for differential expression testing and differetial relative usage testing 
(e.g. alternative splicing, AS and alternative polyadenylation, APA).
"""


def deseq2_normalize(counts_df, sample_list, lowExprGenesQ=0.3, pseudocount = 1):
    """
    Performs DESeq2-style median-of-ratios normalization.
    
    Assumes inputs are validated (sample_list exists in counts_df columns).
    
    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).
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
    # Work on a copy to avoid SettingWithCopy warnings on the original df
    # We only look at the specific samples requested
    df_work = counts_df[sample_list].copy()

    # 1. Calculate Geometric Means
    # We use log-space to prevent overflow: geometric_mean = 2**(mean(log(x)))
    # We only care about genes with >0 counts in ALL samples for the reference
    valid_genes_mask = (df_work > 0).all(axis=1)
    
    # Slice only valid genes
    ref_genes_df = df_work.loc[valid_genes_mask].copy()
    
    # Calculate Log Geometric Mean
    log_counts = np.log2(ref_genes_df)
    log_geom_means = log_counts.mean(axis=1)
    
    # 2. Filter low-expressed genes from Size Factor calculation
    threshold = log_geom_means.quantile(lowExprGenesQ)
    high_expr_genes = log_geom_means[log_geom_means > threshold].index
    
    # 3. Calculate Size Factors (Median of Ratios)
    # Ratio = Count / GeometricMean
    # We use the filtered high-expression set
    ratios = log_counts.loc[high_expr_genes].sub(log_geom_means.loc[high_expr_genes], axis=0)
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
    # Divide original counts by the size factors
    norm_counts = (df_work[sample_list]+pseudocount).div(sfs_df['sf'], axis=1)
    
    return norm_counts, sfs_df