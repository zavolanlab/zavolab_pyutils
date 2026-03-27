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