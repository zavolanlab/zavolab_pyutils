from pathlib import Path
import subprocess

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
    Plots the distribution of Size Factors and Read Sums.
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
    plot_lims=None,
    legend_title="",
    highlight_samples_list=None,
    calculate_permanova_R2=False):
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

    x_feature, y_feature = "PC1", "PC2"
    hue = hue_feature
    hue_order = list(np.sort(metadata_df[hue].unique()))
    palette = list(sns.color_palette(sns_color_palette, len(hue_order)))

    if calculate_permanova_R2 and len(hue_order) > 1:
        R2_AllGenes = np.round(get_MultiDimR2(x,list(principalDf[hue]),True),2)
        R2_PC1andPC2 = np.round(get_MultiDimR2(principalComponents,list(principalDf[hue]),True),2)

    ###
    # PCA-only
    ###

    bw_adjust = 1.75
    alpha_param, s_param = 0.9, 10
    sns.set(font_scale=1.2)
    sns.set_style("white")
    fig, axes = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(5.2, 5.2))

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

    k = 0
    for cat in hue_order:
        ax = sns.kdeplot(
            data=principalDf.loc[principalDf[hue] == cat],
            x=x_feature,
            y=y_feature,
            fill=False,
            levels=[0.25],
            bw_adjust=bw_adjust,
            color=palette[k],
        )
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
        markerscale=2.5,
        ncol=1,
    )

    dir_path = Path(savefig_path).parent

    out = subprocess.check_output(
        "mkdir -p " + str(dir_path),
        shell=True,
    )
    fig.savefig(
        savefig_path,
        bbox_inches="tight",
        dpi=600,
    )