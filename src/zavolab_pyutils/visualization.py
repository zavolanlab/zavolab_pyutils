from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from adjustText import adjust_text
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats

from .read_count_data_analysis import get_MultiDimR2

###
# Diagnostic plots
###

def plot_size_factors(sfs_df, outdir, log_scale=False):
    """
    Plots the diagnostic relationship between library sizes and DESeq2 size factors.

    Generates a scatter plot with a calculated Spearman correlation to verify 
    normalization behavior. Ideally, size factors should scale linearly with 
    sequencing depth.

    Parameters
    ----------
    sfs_df : pandas.DataFrame
        DataFrame containing size factors and read sums. Must include the 
        columns 'sf' (size factor) and 'read_sum_mln' (millions of reads).
    outdir : str or pathlib.Path
        Directory path where the generated plots (.png and .pdf) will be saved.
    log_scale : bool, optional
        If True, applies a log2 transformation to both axes. A small pseudocount 
        is added to read sums to prevent log(0). Default is False.

    Returns
    -------
    None
        Saves 'library_size_vs_SF.png' and 'library_size_vs_SF.pdf' to `outdir`.
    """
    data = sfs_df.copy()
    if log_scale:
        data["read_sum_mln"] = np.log2(data["read_sum_mln"] + 10**(-6))
        data["sf"] = np.log2(data["sf"] + 1)

    alpha_param, s_param = 0.7, 40
    sns.set(font_scale=1)
    sns.set_style("white")
    fig, axes = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(4, 4))
    
    x_feature, y_feature = "read_sum_mln", "sf"
    
    ax = sns.scatterplot(
        ax=axes,
        data=data,
        x=x_feature,
        y=y_feature,
        s=s_param,
        alpha=alpha_param,
        edgecolor="black",
        linewidth=0.5,
        color='teal',
    )

    spearman_corr = stats.spearmanr(a=data[x_feature],b=data[y_feature])[0]
    
    xlabel = "# reads, mln"
    ylabel = "size factor,\nDeseq2 normalization"
    if log_scale:
        xlabel = xlabel + ", log2"
        ylabel = ylabel + ", log2"
    
    ax.set(xlabel=xlabel,
           ylabel=ylabel,
          title="spearman corr = "+str(np.round(spearman_corr,2)))
    ax.tick_params(left=True, bottom=True)
    
    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(
                os.path.join(outdir, "library_size_vs_SF.png"),
                bbox_inches="tight",
                dpi=600
            )
            fig.savefig(
                os.path.join(outdir, "library_size_vs_SF.pdf"),
                bbox_inches="tight",
                dpi=600,
            )
        except Exception as e:
            print(f"Error saving plot: {e}")


def pca_plot(
    data_df,
    samples_list,
    metadata_df,
    hue_feature,
    savefig_path,
    sns_color_palette="hls",
    hue_order = None,
    plot_lims=None,
    legend_title="",
    highlight_samples_list=None,
    calculate_permanova_R2=False,
    add_2D_KDE_countours=False,
    bw_adjust=1.75,alpha_param=0.9, s_param=10,figsize=(5.2,5.2),legend_markerscale=1.5):
    """
    Performs Principal Component Analysis (PCA) and generates a customized 2D scatter plot.

    Standardizes the input features (genes) across samples, computes the first two 
    principal components, and plots the samples colored by a specified metadata categorical 
    feature. Optionally draws KDE density contours and calculates PERMANOVA R2 statistics 
    to quantify group separation.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Gene expression matrix with genes as rows and samples as columns.
    samples_list : list of str
        List of column names in `data_df` corresponding to the samples to be plotted.
    metadata_df : pandas.DataFrame
        Metadata mapping. Must contain a 'sample' column matching `samples_list`, 
        and a column matching the string passed to `hue_feature`.
    hue_feature : str
        The column name in `metadata_df` used to group and color the samples.
    savefig_path : str or pathlib.Path
        Full file path (including filename and extension) where the plot will be saved.
    sns_color_palette : str, optional
        Seaborn color palette name to use for the categorical groups. Default is "hls".
    plot_lims : tuple of tuple, optional
        Axis limits formatted as ((xmin, xmax), (ymin, ymax)). Default is None (auto-scale).
    legend_title : str, optional
        Title for the plot legend. Default is an empty string "".
    highlight_samples_list : list of str, optional
        List of specific sample names to highlight with enlarged markers. Default is None.
    calculate_permanova_R2 : bool, optional
        If True, calculates and displays the PERMANOVA R2 values in the plot title 
        (both for the full scaled dataset and the 2D PCA projection). Default is False.

    Returns
    -------
    None
        Saves the PCA plot to the specified `savefig_path`.
    """
    # assumes that data_df is gene expression matrix with genes in rows and samples in columns
    # and may be having additional columns like "gene length" etc
    # therefore, "samples_list" should match column names of data_df
    # metadata_df should have a "sample" column matching the sample names in data_df, and a "hue_feature" column with categorical values for coloring the PCA plot

    x = (data_df[samples_list].values).transpose()
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])
    principalDf["sample"] = samples_list
    principalDf = pd.merge(principalDf, metadata_df, how="left", on="sample") # 

    # fill NA values in hue_feature with "NA" string and convert to string type for consistent coloring in seaborn
    if principalDf[hue_feature].isna().any(): 
        raise ValueError(
            """NA values found in hue_feature column. 
            Please fill NA values with a string (e.g. 'NA') before plotting."""
        )

    x_feature, y_feature = "PC1", "PC2"
    hue = hue_feature
    if hue_order is None:
        hue_order = list(np.sort(principalDf[hue].dropna().unique()))
    palette = list(sns.color_palette(sns_color_palette, len(hue_order)))

    if calculate_permanova_R2 and len(hue_order) > 1:
        R2_AllGenes = np.round(get_MultiDimR2(x,list(principalDf[hue]),True),2)
        R2_PC1andPC2 = np.round(get_MultiDimR2(principalComponents,list(principalDf[hue]),True),2)

    ###
    # PCA-only
    ###

    sns.set(font_scale=1.2)
    sns.set_style("white")
    fig, axes = plt.subplots(1, 1, sharey=False, sharex=False, figsize=figsize)

    ax = sns.scatterplot(
        data=principalDf,
        x=x_feature,
        y=y_feature,
        s=s_param,
        alpha=alpha_param,
        edgecolor="black",
        linewidth=0.3,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
    )
    if highlight_samples_list is not None:
        highlight_principalDf = principalDf.loc[
            principalDf["sample"].isin(highlight_samples_list)
        ].reset_index(drop=True)
        ax = sns.scatterplot(
            data=highlight_principalDf,
            x=x_feature,
            y=y_feature,
            s=s_param * 10,
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
            hue=hue,
            hue_order=hue_order,
            palette=palette,
            legend=False,
        )
    if add_2D_KDE_countours:
        k = 0
        for cat in hue_order:
            cat_data = principalDf.loc[principalDf[hue] == cat]
            if len(cat_data) >= 3:
                ax = sns.kdeplot(
                    data=cat_data,
                    x=x_feature,
                    y=y_feature,
                    fill=False,
                levels=[0.25],
                bw_adjust=bw_adjust,
                color=palette[k])
            k = k + 1

    ax.set(
        xlabel="PC1, "
        + str(int(np.round(pca.explained_variance_ratio_[0] * 100, 0)))
        + "% variance"
    )
    ax.set(
        ylabel="\nPC2, "
        + str(int(np.round(pca.explained_variance_ratio_[1] * 100, 0)))
        + "% variance"
    )
    if calculate_permanova_R2 and len(hue_order) > 1:
        ax.set(title=str(len(data_df)) + " top expressed genes"+\
               "\nPERMANOVA R2 (all genes) = "+str(R2_AllGenes)+\
                "\nPERMANOVA R2 (PC1&PC2) = "+str(R2_PC1andPC2))
    else:
        ax.set(title=str(len(data_df)) + " top expressed genes")
    
    if plot_lims is not None:
        ax.set(xlim=plot_lims[0], ylim=plot_lims[1])
    ax.tick_params(bottom=True, left=True)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        title=legend_title,
        markerscale=legend_markerscale,
        ncol=1,
    )

    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        savefig_path,
        bbox_inches="tight",
        dpi=600,
    )

def plot_gene_expression_with_ci(
    norm_counts_df, metadata_df, selected_genes, regr_model_df, 
    savefig_path, sample_col='sample', cond_col='condition', 
    CI_limit=0.95, selected_CI_model='QuantReg', gene_col=None
):
    """
    Plots log2 expression of selected genes with empirical Bayes confidence intervals,
    calculated dynamically using the exact Negative Binomial distribution.
    
    Parameters
    ----------
    norm_counts_df : pandas.DataFrame
        Normalized count matrix. Rows should correspond to genes, and columns to samples. 
        May contain additional non-sample columns (e.g. gene length) which will be ignored.
    metadata_df : pandas.DataFrame
        Metadata mapping samples to conditions.
    selected_genes : list of str
        List of genes to plot.
    regr_model_df : pandas.DataFrame
        Model parameters generated by `model_mean_variance`.
    savefig_path : str or pathlib.Path
        Output path for the figure.
    sample_col : str, optional
        Column in metadata_df containing sample names. Default is 'sample'.
    cond_col : str, optional
        Column in metadata_df containing condition labels. Default is 'condition'.
    CI_limit : float, optional
        Confidence interval limit (e.g. 0.95 for 95%). Default is 0.95.
    selected_CI_model : str, optional
        Model type to use from `regr_model_df`. Default is 'QuantReg'.
    gene_col : str, optional
        Column name in `norm_counts_df` containing gene identifiers. 
        If None, the DataFrame index is used. Default is None.
    """
    df_work = norm_counts_df.copy()
    
    # 1. Identify genes and standardize the gene column name to 'gene_name'
    if gene_col is not None:
        if gene_col not in df_work.columns:
            raise ValueError(f"Column '{gene_col}' not found in norm_counts_df.")
        common_genes = [g for g in selected_genes if g in df_work[gene_col].values]
        data = df_work[df_work[gene_col].isin(common_genes)].copy()
        data = data.rename(columns={gene_col: 'gene_name'})
    else:
        common_genes = [g for g in selected_genes if g in df_work.index]
        idx_name = df_work.index.name if df_work.index.name else 'index'
        data = df_work.loc[common_genes].reset_index().rename(columns={idx_name: 'gene_name'})
        
    if data.empty:
        print("Warning: None of the selected genes were found in the dataset.")
        return

    # 2. VALIDATION: Check for duplicate rows for any selected gene
    duplicate_genes = data['gene_name'].value_counts()
    duplicates = duplicate_genes[duplicate_genes > 1].index.tolist()
    if duplicates:
        raise ValueError(
            f"Multiple rows found for the following selected genes: {duplicates}. "
            "Please aggregate or filter your data so each gene has only one row before plotting."
        )

    # 3. Safely melt ONLY the sample columns
    valid_samples = [s for s in metadata_df[sample_col] if s in data.columns]
    melted = pd.melt(data, id_vars=['gene_name'], value_vars=valid_samples, var_name=sample_col, value_name='expr')
    
    melted['log2_expr'] = np.log2(melted['expr'] + 1e-6)
    melted = pd.merge(metadata_df[[sample_col, cond_col]], melted, how='right', on=sample_col)
    
    order = sorted(melted[cond_col].unique())
    
    # Calculate limits adjusted for multiple conditions
    N_conditions = len(order)
    adjCI_limit = 1 - (1 - CI_limit) / N_conditions
    Low_q = (1 - adjCI_limit) / 2
    Up_q = 1 - Low_q
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(common_genes), sharey=True, sharex=False, figsize=(2.8*len(common_genes), 5.2))
    if len(common_genes) == 1: axes = [axes]
        
    for k, gene in enumerate(common_genes):
        ax = axes[k]
        gene_data = melted[melted['gene_name'] == gene]
        
        summary_list = []
        for cond in order:
            cond_data = gene_data[gene_data[cond_col] == cond]['expr'].values
            if len(cond_data) == 0: continue
                
            M = np.mean(cond_data)
            log2_M = np.log2(max(M, 2**(-18)))
            n_reps = len(cond_data)
            
            # Lookup dispersion alpha for this condition
            cond_params = regr_model_df[(regr_model_df['condition'] == cond) & (regr_model_df['model_type'] == selected_CI_model)]
            
            if not cond_params.empty:
                alpha = cond_params.loc[cond_params['pred_feature'] == 'var', 'param'].iloc[0]
                alpha = max(alpha, 1e-6) # Ensure alpha is positive for NB distribution
                
                # Setup Negative Binomial parameters for the sum of 'n_reps' samples
                n_param = n_reps / alpha
                p_param = 1 / (1 + alpha * M)
                
                # Calculate Exact NB Quantiles for the sum, then divide by n_reps for the mean
                LB_sum = stats.nbinom.ppf(Low_q, n_param, p_param)
                UB_sum = stats.nbinom.ppf(Up_q, n_param, p_param)
                
                LB = max(1e-3, LB_sum / n_reps)
                UB = max(1e-3, UB_sum / n_reps)
            else:
                LB, UB = M, M # Fallback
                
            summary_list.append({
                'condition': cond,
                'log2_M': log2_M,
                'err_lower': log2_M - np.log2(LB),
                'err_upper': np.log2(UB) - log2_M
            })
            
        summary_df = pd.DataFrame(summary_list)
        y_pos = np.arange(len(order))
        
        ax.plot(summary_df['log2_M'], y_pos, color='grey', zorder=1, alpha=0.7)
        ax.errorbar(
            summary_df['log2_M'], y_pos, 
            xerr=[summary_df['err_lower'], summary_df['err_upper']], 
            fmt='o', color='black', capsize=4, zorder=2, markersize=5
        )
        sns.stripplot(
            ax=ax, data=gene_data, x='log2_expr', y=cond_col, 
            order=order, color='white', size=4, edgecolor='black', 
            linewidth=1, alpha=0.5, zorder=3, jitter=True
        )
        
        ax.set(title=gene, ylabel='', xlabel='$log_2~norm. expr$')
        if k > 0: ax.tick_params(left=False)
        
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_mean_variance_diagnostics(all_plot_data, savefig_path):
    """
    Plots the diagnostic regression fits for variance within each condition.
    """
    conditions = all_plot_data['condition'].unique()
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(conditions), sharey=True, sharex=True, figsize=(5 * len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]
        
    lims = (-50, all_plot_data['mean'].quantile(0.99))
    
    for i, cond in enumerate(conditions):
        ax = axes[i]
        data = all_plot_data[all_plot_data['condition'] == cond]
        
        sns.scatterplot(
            ax=ax, data=data.sample(min(3000, len(data))), 
            x='mean', y='var', s=5, alpha=0.3
        )
        
        sns.lineplot(
            ax=ax, data=data, 
            y='QuantReGpred_var', x='mean', 
            color='blue', label='QuantReg'
        )
        
        diag_df = pd.DataFrame([[lims[0], lims[0]], [lims[1], lims[1]]], columns=['mean', 'var'])
        sns.lineplot(ax=ax, data=diag_df, y='var', x='mean', color='red', label='diagonal')
        
        ax.set(xlabel='mean gene expression', ylabel='empirical variance', title=cond, xlim=lims, ylim=lims)
        ax.tick_params(bottom=True, left=(i==0))
        
    fig.tight_layout(pad=0.3)
    
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)


def plot_mean_vs_cv(
    norm_counts_df, metadata_df, savefig_path, 
    sample_col='sample', cond_col='condition', is_log2=False
):
    """
    Plots Mean Expression vs Coefficient of Variation (CV).
    
    Systematic correlations between CVs and the mean of normalized expression 
    levels reflect to what extent a normalization method has failed to correct 
    for Poisson sampling noise. Ideally, the correlation should be near zero.
    """
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = norm_counts_df.columns.intersection(sample_map.index)
    
    df_work = norm_counts_df[common_samples].copy()
    if is_log2:
        df_work = 2 ** df_work # Convert to linear scale for standard CV calculation
        
    conditions = sample_map[common_samples].unique()
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))
    if len(conditions) == 1: axes = [axes]
        
    for i, cond in enumerate(conditions):
        samples_in_cond = sample_map[sample_map == cond].index
        if len(samples_in_cond) < 2: continue
            
        cond_data = df_work[samples_in_cond].values
        means = np.mean(cond_data, axis=1)
        stds = np.std(cond_data, axis=1, ddof=1)
        
        # Filter zero means
        mask = means > 0
        means, stds = means[mask], stds[mask]
        cvs = stds / means
        
        # Calculate correlations on log10 values to prevent skewing by extreme outliers
        log_means = np.log10(means + 1e-6)
        log_cvs = np.log10(cvs + 1e-6)
        
        pearson_r, _ = stats.pearsonr(log_means, log_cvs)
        spearman_r, _ = stats.spearmanr(means, cvs)
        
        ax = axes[i]
        sns.scatterplot(x=means, y=cvs, ax=ax, s=5, alpha=0.3, color='teal')
        ax.set(
            xscale='log', yscale='log', 
            xlabel='Mean Expression', ylabel='Coefficient of Variation (CV)',
            title=f"{cond}\nPearson: {pearson_r:.2f} | Spearman: {spearman_r:.2f}"
        )
        
    fig.tight_layout()
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_sanity_gene_expression_with_ci(
    sample_norm_df, means_df, errors_df, metadata_df, selected_genes, 
    savefig_path, sample_col='sample', cond_col='condition', CI_limit=0.95
):
    """
    Plots Sanity log2 normalized counts with Bayesian 95% CI error bars.
    """
    common_genes = [g for g in selected_genes if g in sample_norm_df.index]
    
    melted = sample_norm_df.loc[common_genes].reset_index().rename(columns={'index': 'gene_name'})
    melted = pd.melt(melted, id_vars=['gene_name'], var_name=sample_col, value_name='log2_expr')
    melted = pd.merge(metadata_df[[sample_col, cond_col]], melted, how='right', on=sample_col)
    
    order = sorted(melted[cond_col].unique())
    z_score = stats.norm.ppf(1 - (1 - CI_limit) / 2) # e.g., 1.96 for 95% CI
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(common_genes), sharey=True, figsize=(2.8*len(common_genes), 5.2))
    if len(common_genes) == 1: axes = [axes]
        
    for k, gene in enumerate(common_genes):
        ax = axes[k]
        gene_data = melted[melted['gene_name'] == gene]
        
        # Prepare condition statistics
        y_pos = np.arange(len(order))
        log2_means = means_df.loc[gene, order].values
        err_margins = errors_df.loc[gene, order].values * z_score
        
        ax.plot(log2_means, y_pos, color='grey', zorder=1, alpha=0.7)
        ax.errorbar(
            log2_means, y_pos, xerr=err_margins, 
            fmt='o', color='black', capsize=4, zorder=2, markersize=5
        )
        sns.stripplot(
            ax=ax, data=gene_data, x='log2_expr', y=cond_col, 
            order=order, color='white', size=4, edgecolor='black', 
            linewidth=1, alpha=0.5, zorder=3, jitter=True
        )
        
        ax.set(title=gene, ylabel='', xlabel='$log_2~expr$')
        if k > 0: ax.tick_params(left=False)
        
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_expr_vs_libsize_correlation(
    raw_counts_df, norm_counts_df, metadata_df, savefig_path, 
    sample_col='sample', cond_col='condition', method='spearman', separate_conditions=False
):
    """
    Plots the histogram of correlations between normalized gene expression 
    and raw sample library sizes. 
    """
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = norm_counts_df.columns.intersection(sample_map.index)
    
    X_norm = norm_counts_df[common_samples].values
    y_lib_sizes = raw_counts_df[common_samples].sum(axis=0).values
    conditions = sample_map[common_samples].values
    
    # Fast Vectorized Correlation calculation
    def calc_corr(X, y, corr_method):
        if corr_method == 'spearman':
            X = stats.rankdata(X, axis=1)
            y = stats.rankdata(y)
        X_m = X - np.mean(X, axis=1, keepdims=True)
        y_m = y - np.mean(y)
        cov = np.sum(X_m * y_m, axis=1)
        var_X = np.sum(X_m**2, axis=1)
        var_y = np.sum(y_m**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            return cov / np.sqrt(var_X * var_y)

    sns.set(font_scale=1, style="white")
    
    if separate_conditions:
        unique_conds = np.unique(conditions)
        fig, axes = plt.subplots(1, len(unique_conds), figsize=(5 * len(unique_conds), 4), sharey=True)
        if len(unique_conds) == 1: axes = [axes]
        
        for i, cond in enumerate(unique_conds):
            mask = conditions == cond
            corrs = calc_corr(X_norm[:, mask], y_lib_sizes[mask], method)
            corrs = corrs[~np.isnan(corrs)] # Drop NaNs (genes with zero variance)
            
            sns.histplot(corrs, ax=axes[i], bins=50, color='royalblue', kde=True)
            axes[i].set(title=f"{cond} (Median: {np.median(corrs):.2f})", xlabel=f'{method.title()} Correlation', xlim=(-1, 1))
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        corrs = calc_corr(X_norm, y_lib_sizes, method)
        corrs = corrs[~np.isnan(corrs)]
        
        sns.histplot(corrs, ax=ax, bins=50, color='royalblue', kde=True)
        ax.set(title=f"All Samples (Median: {np.median(corrs):.2f})", xlabel=f'{method.title()} Correlation', xlim=(-1, 1))
        
    fig.tight_layout()
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_variance_vs_expression(means_df, vg_df, savefig_path, true_vg=None):
    """
    Plots the inferred biological variance (v_g) against mean log2 expression.
    Highlights how the algorithm treats lowly vs. highly expressed genes.
    """
    # Calculate approximate base expression across all conditions
    base_expr = means_df.mean(axis=1)
    
    plot_df = pd.DataFrame({
        'log2_expr': base_expr,
        'v_g': vg_df['inferred_v_g']
    })
    
    sns.set(font_scale=1, style="white")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.lineplot(data=plot_df, x='log2_expr', y='v_g', ax=ax, color='teal')
    
    if true_vg is not None:
        ax.axhline(true_vg, color='red', linestyle='--', label=f'True Simulated v_g ({true_vg})')
        ax.legend()
        
    ax.set(
        title="Inferred Biological Variance vs. Mean Expression",
        xlabel="Mean Log2 Expression",
        ylabel="Inferred Biological Variance (v_g)"
    )
    
    fig.tight_layout()
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)