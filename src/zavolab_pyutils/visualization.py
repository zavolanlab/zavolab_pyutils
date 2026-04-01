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

def plot_size_factors(sfs_df, savefig_path, log_scale=False):
    """
    Plots the diagnostic relationship between library sizes and DESeq2 size factors.

    Parameters
    ----------
    sfs_df : pandas.DataFrame
        DataFrame containing size factors and read sums. 
    savefig_path : str or pathlib.Path
        Output path where the generated plot will be saved.
    log_scale : bool, optional
        If True, applies a log2 transformation to both axes. Default is False.
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
    
    if savefig_path:
        try:
            # Create parent directories if they do not exist
            dir_path = Path(savefig_path).parent
            dir_path.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(
                savefig_path,
                bbox_inches="tight",
                dpi=600
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
    norm_counts_df, means_df, errors_df, metadata_df, selected_genes, 
    savefig_path, sample_col='sample', cond_col='condition', 
    CI_limit=0.95, adjust_multiple_comparisons=False, log_scale=True
):
    """
    Plots DESeq2 normalized linear counts with modeled CI error bars.
    
    Parameters
    ----------
    norm_counts_df : pd.DataFrame
        Normalized count matrix (linear scale).
    means_df, errors_df : pd.DataFrame
        Means and SEMs output from `get_deseq2_means_and_errors`.
    metadata_df : pd.DataFrame
        Metadata mapping samples to conditions.
    selected_genes : list of str
        List of genes to plot.
    savefig_path : str or pathlib.Path
        Output file path.
    adjust_multiple_comparisons : bool, optional
        If True, applies a Bonferroni correction to the CI width based on the 
        number of pairwise condition comparisons. Default is False.
    log_scale : bool, optional
        If True, visualizes the linear data on a log10 x-axis. Default is True.
    """
    common_genes = [g for g in selected_genes if g in norm_counts_df.index]
    melted = norm_counts_df.loc[common_genes].reset_index().rename(columns={'index': 'gene_name'})
    melted = pd.melt(melted, id_vars=['gene_name'], var_name=sample_col, value_name='expr')
    melted = pd.merge(metadata_df[[sample_col, cond_col]], melted, how='right', on=sample_col)
    
    order = sorted(melted[cond_col].unique())
    n_conditions = len(order)
    
    # Calculate Alpha with optional Bonferroni correction
    alpha_val = 1.0 - CI_limit
    if adjust_multiple_comparisons and n_conditions > 2:
        num_comparisons = (n_conditions * (n_conditions - 1)) / 2
        alpha_val /= num_comparisons
        
    z_score = stats.norm.ppf(1 - alpha_val / 2)
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(common_genes), sharey=True, figsize=(2.8*len(common_genes), 5.2))
    if len(common_genes) == 1: axes = [axes]
        
    for k, gene in enumerate(common_genes):
        ax = axes[k]
        gene_data = melted[melted['gene_name'] == gene]
        y_pos = np.arange(len(order))
        
        means = means_df.loc[gene, order].values
        err_margins = errors_df.loc[gene, order].values * z_score
        
        ax.plot(means, y_pos, color='grey', zorder=1, alpha=0.7)
        ax.errorbar(means, y_pos, xerr=err_margins, fmt='o', color='black', capsize=4, zorder=2, markersize=5)
        sns.stripplot(
            ax=ax, data=gene_data, x='expr', y=cond_col, order=order, 
            color='white', size=4, edgecolor='black', linewidth=1, alpha=0.5, zorder=3, jitter=True
        )
        
        if log_scale:
            ax.set_xscale('log', base=2)
            ax.set_xlabel('$log_2$ expr')
        else:
            ax.set_xlabel('expr')
            
        ax.set(title=gene, ylabel='')
        if k > 0: ax.tick_params(left=False)
        
    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
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
    
    Parameters
    ----------
    norm_counts_df : pd.DataFrame
        Normalized count matrix.
    metadata_df : pd.DataFrame
        Metadata mapping samples to conditions.
    savefig_path : str or pathlib.Path
        Output file path.
    sample_col : str, optional
        Column name in metadata_df. Default is 'sample'.
    cond_col : str, optional
        Column name in metadata_df. Default is 'condition'.
    is_log2 : bool, optional
        Set to True if norm_counts_df is in log2 scale (e.g., Sanity output). 
        Set to False for linear scale (e.g., DESeq2). Default is False.
        
    Returns
    -------
    plot_data_df : pd.DataFrame
        A long-format DataFrame containing 'gene', 'condition', 'log2_mean', and 'log10_cv'.
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
        
    plot_data_list = [] # List to collect the dataframes for each condition
        
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
        valid_genes = df_work.index[mask] # Capture the gene names that survived the filter
        
        # Calculate correlations on log2 values to prevent skewing by extreme outliers
        log2_means = np.log2(means + 1e-6)
        log_cvs = np.log10(cvs + 1e-6)
        
        pearson_r, _ = stats.pearsonr(log2_means, log_cvs)
        spearman_r, _ = stats.spearmanr(log2_means, log_cvs)
        
        # Store data for this condition
        cond_df = pd.DataFrame({
            'gene': valid_genes,
            'condition': cond,
            'log2_mean': log2_means,
            'log10_cv': log_cvs
        })
        plot_data_list.append(cond_df)
        
        ax = axes[i]
        sns.scatterplot(x=log2_means, y=log_cvs, ax=ax, s=5, alpha=0.3, color='teal')
        ax.set( 
            xlabel='Mean $log_2$ Expression', ylabel='Coefficient of Variation (CV), $log_{10}$',
            title=f"{cond}\nPearson: {pearson_r:.2f} | Spearman: {spearman_r:.2f}"
        )
        ax.tick_params(left=True, bottom=True)
        
    fig.tight_layout()
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)
    
    # Combine all condition dataframes into one and return
    if plot_data_list:
        final_df = pd.concat(plot_data_list, ignore_index=True)
    else:
        final_df = pd.DataFrame()
        
    return final_df

def plot_sanity_gene_expression_with_ci(
    sample_norm_df, means_df, errors_df, metadata_df, selected_genes, 
    savefig_path, sample_col='sample', cond_col='condition', 
    CI_limit=0.95, adjust_multiple_comparisons=False
):
    """
    Plots Sanity log2 normalized counts with Bayesian CI error bars.
    
    Parameters
    ----------
    sample_norm_df, means_df, errors_df : pd.DataFrame
        Outputs directly from `apply_sanity_normalization`.
    metadata_df : pd.DataFrame
        Metadata mapping samples to conditions.
    selected_genes : list of str
        List of genes to plot.
    savefig_path : str or pathlib.Path
        Output file path.
    adjust_multiple_comparisons : bool, optional
        If True, applies a Bonferroni correction to the CI width based on the 
        number of pairwise condition comparisons. Default is False.
    """
    
    input_data_df = sample_norm_df.copy()
    input_data_df.index.name = 'index'
    
    common_genes = [g for g in selected_genes if g in input_data_df.index]

    melted = input_data_df.loc[common_genes].reset_index().rename(columns={'index': 'gene_name'})
    melted = pd.melt(melted, id_vars=['gene_name'], var_name=sample_col, value_name='log2_expr')
    melted = pd.merge(metadata_df[[sample_col, cond_col]], melted, how='right', on=sample_col)
    
    order = sorted(melted[cond_col].unique())
    n_conditions = len(order)
    
    # Calculate Alpha with optional Bonferroni correction
    alpha_val = 1.0 - CI_limit
    if adjust_multiple_comparisons and n_conditions > 2:
        num_comparisons = (n_conditions * (n_conditions - 1)) / 2
        alpha_val /= num_comparisons
        
    z_score = stats.norm.ppf(1 - alpha_val / 2)
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(common_genes), sharey=True, figsize=(2.8*len(common_genes), 5.2))
    if len(common_genes) == 1: axes = [axes]
        
    for k, gene in enumerate(common_genes):
        ax = axes[k]
        gene_data = melted[melted['gene_name'] == gene]
        y_pos = np.arange(len(order))
        
        log2_means = means_df.loc[gene, order].values
        err_margins = errors_df.loc[gene, order].values * z_score
        
        ax.plot(log2_means, y_pos, color='grey', zorder=1, alpha=0.7)
        ax.errorbar(log2_means, y_pos, xerr=err_margins, fmt='o', color='black', capsize=4, zorder=2, markersize=5)
        sns.stripplot(
            ax=ax, data=gene_data, x='log2_expr', y=cond_col, order=order, 
            color='white', size=4, edgecolor='black', linewidth=1, alpha=0.5, zorder=3, jitter=True
        )
        
        ax.set(title=gene, ylabel='', xlabel='$log_2~expr$')
        ax.tick_params(left=True, bottom=True)
        if k > 0: ax.tick_params(left=False)
        
    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_expr_vs_libsize_correlation(
    raw_counts_df, norm_counts_df, metadata_df, savefig_path, 
    sample_col='sample', cond_col='condition', method='spearman', separate_conditions=False
):
    """
    Plots the histogram of correlations between normalized expression and library sizes.
    
    Parameters
    ----------
    raw_counts_df : pd.DataFrame
        Unnormalized count matrix used to calculate library depth.
    norm_counts_df : pd.DataFrame
        Normalized count matrix.
    metadata_df : pd.DataFrame
        Metadata mapping samples to conditions.
    savefig_path : str or pathlib.Path
        Output file path.
    sample_col, cond_col : str, optional
        Column names in metadata.
    method : str, optional
        Correlation method ('spearman' or 'pearson'). Default is 'spearman'.
    separate_conditions : bool, optional
        If True, plots individual histograms per condition. Default is False.
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
    Plots the inferred biological variance (v_g) against mean log2 expression (Sanity diagnostics).
    
    Parameters
    ----------
    means_df : pd.DataFrame
        Estimated mean log2 expression level per condition.
    vg_df : pd.DataFrame
        Inferred biological variance (v_g) for each gene.
    savefig_path : str or pathlib.Path
        Output file path.
    true_vg : float, optional
        Value to plot as a horizontal true reference line (for simulated data).
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
    ax.tick_params(left=True, bottom=True)
    
    fig.tight_layout()
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_sanity_relative_usage_with_ci(
    norm_counts_df, variances_df, metadata_df, isoform_pairs, 
    savefig_path, sample_col='sample', cond_col='condition', 
    CI_limit=0.95, adjust_multiple_comparisons=False,
    log2_scale=True,
):
    """
    Plots Sanity relative usage (ratio) of isoform pairs with Bayesian CI error bars.
    
    Parameters
    ----------
    norm_counts_df : pd.DataFrame
        Cell-level log2 expression estimates (output from Sanity).
    variances_df : pd.DataFrame
        Cell-level log2 posterior variances (output from Sanity).
    metadata_df : pd.DataFrame
        Metadata mapping samples to conditions.
    isoform_pairs : list of tuples
        List of (Isoform_1_ID, Isoform_2_ID) to compare. (e.g., [('GeneX_Proximal', 'GeneX_Distal')])
    savefig_path : str or pathlib.Path
        Output file path.
    sample_col : str, optional
        Column in metadata_df with sample IDs. Default is 'sample'.
    cond_col : str, optional
        Column in metadata_df with condition labels. Default is 'condition'.
    CI_limit : float, optional
        Confidence interval limit (e.g., 0.95 for 95% CI). Default is 0.95.
    adjust_multiple_comparisons : bool, optional
        If True, applies a Bonferroni correction to the CI width based on the 
        number of pairwise condition comparisons. Default is False.
    log2_scale : bool, optional
        If True (default), plots values on the log2 scale. 
        If False, values and error bars are exponentiated to the natural ratio scale.
    """
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    order = sorted(sample_map.dropna().unique())
    n_conditions = len(order)
    
    # Calculate Alpha with optional Bonferroni correction
    alpha_val = 1.0 - CI_limit
    if adjust_multiple_comparisons and n_conditions > 2:
        num_comparisons = (n_conditions * (n_conditions - 1)) / 2
        alpha_val /= num_comparisons
        
    z_score = stats.norm.ppf(1 - alpha_val / 2)
    
    sns.set(font_scale=1, style="white")
    
    # Filter valid pairs to ensure both isoforms exist in the count matrix
    valid_pairs = []
    for iso1, iso2 in isoform_pairs:
        if iso1 in norm_counts_df.index and iso2 in norm_counts_df.index:
            valid_pairs.append((iso1, iso2))
        else:
            print(f"Warning: One or both isoforms not found in count matrix for pair ({iso1}, {iso2}). Skipping this pair.")
            
    if not valid_pairs:
        print("No valid isoform pairs found in the provided data.")
        return
        
    # Set up matplotlib figure dimensions
    fig, axes = plt.subplots(1, len(valid_pairs), sharey=True, figsize=(2.8*len(valid_pairs), 5.2))
    if len(valid_pairs) == 1: 
        axes = [axes]
        
    for k, (iso1, iso2) in enumerate(valid_pairs):
        ax = axes[k]
        pair_name = f"{iso1}\nvs\n{iso2}"
        
        # 1. Calculate cell-level log2 ratios and aggregate posterior variances
        log2_ratio_cells = norm_counts_df.loc[iso1] - norm_counts_df.loc[iso2]
        var_ratio_cells = variances_df.loc[iso1] + variances_df.loc[iso2]
        
        # Prepare data for seaborn stripplot
        pair_data = pd.DataFrame({
            sample_col: log2_ratio_cells.index,
            'log2_ratio': log2_ratio_cells.values
        })
        pair_data = pd.merge(metadata_df[[sample_col, cond_col]], pair_data, how='inner', on=sample_col)
        
        y_pos = np.arange(len(order))
        log2_means = []
        err_margins = []
        
        # 2. Calculate condition-level aggregated stats for the error bars
        for cond in order:
            cells_in_cond = sample_map[sample_map == cond].index.intersection(norm_counts_df.columns)
            n_cells = len(cells_in_cond)
            
            if n_cells == 0:
                log2_means.append(np.nan)
                err_margins.append(np.nan)
                continue
                
            # Mean ratio
            mean_ratio = log2_ratio_cells[cells_in_cond].mean()
            
            # Combine empirical variance and marginalized posterior variance
            empirical_var = log2_ratio_cells[cells_in_cond].var(ddof=1) if n_cells > 1 else 0
            posterior_var = var_ratio_cells[cells_in_cond].mean()
            
            se_ratio = np.sqrt((empirical_var + posterior_var) / n_cells)
            
            log2_means.append(mean_ratio)
            err_margins.append(se_ratio * z_score)
            
        log2_means = np.array(log2_means)
        err_margins = np.array(err_margins)
        
        # --- NEW LOGIC: Adjust for scale ---
        if log2_scale:
            plot_means = log2_means
            xerr = err_margins
            x_col = 'log2_ratio'
            x_label = '$log_2$(rel. usage ratio)'
        else:
            # Exponentiate the raw cell data for the stripplot
            pair_data['natural_ratio'] = 2 ** pair_data['log2_ratio']
            x_col = 'natural_ratio'
            x_label = 'Relative usage ratio'
            
            # Exponentiate the means and CI bounds
            plot_means = 2 ** log2_means
            lower_bounds = 2 ** (log2_means - err_margins)
            upper_bounds = 2 ** (log2_means + err_margins)
            
            # Matplotlib asymmetric errors: distance from mean to lower bound, and mean to upper bound
            xerr = np.array([
                plot_means - lower_bounds, 
                upper_bounds - plot_means
            ])
            
        # 3. Plotting
        # Draw the lines connecting condition means
        ax.plot(plot_means, y_pos, color='grey', zorder=1, alpha=0.7)
        
        # Draw the error bars representing the Bayesian Confidence Intervals
        ax.errorbar(plot_means, y_pos, xerr=xerr, fmt='o', color='black', capsize=4, zorder=2, markersize=5)
        
        # Overlay the individual cell points
        sns.stripplot(
            ax=ax, data=pair_data, x=x_col, y=cond_col, order=order, 
            color='white', size=4, edgecolor='black', linewidth=1, alpha=0.5, zorder=3, jitter=True
        )
        
        # Formatting
        ax.set(title=pair_name, ylabel='', xlabel=x_label)
        ax.tick_params(left=True, bottom=True)
        if k > 0: 
            ax.tick_params(left=False)
        
    # Save the figure
    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)