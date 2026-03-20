import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
import numpy as np
from adjustText import adjust_text
from scipy import stats

###
# Diagnostic plots
###

def plot_size_factors(sfs_df, meta_df, outdir):
    """
    Plots the distribution of Size Factors and Read Sums.
    """
    data = pd.merge(sfs_df.copy().reset_index(drop=True), 
                    meta_df.copy().reset_index(drop=True), 
                    how="left", on="sample")
    
    alpha_param, s_param = 0.7, 40
    sns.set(font_scale=1.2)
    sns.set_style("white")
    fig, axes = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(4, 4))
    
    x_feature, y_feature = "read_sum_mln", "sf"
    
    hue = "fraction"
    hue_order = ["F", "T"]
    palette = ["orange", "teal"]
    
    ax = sns.scatterplot(
        ax=axes,
        data=data,
        x=x_feature,
        y=y_feature,
        s=s_param,
        alpha=alpha_param,
        edgecolor="black",
        linewidth=0.5,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
    )

    spearman_corr = stats.spearmanr(a=data[x_feature],b=data[y_feature])[0]
    
    ax.set(xlabel="# reads, mln",
           ylabel="size factor,\nDeseq2 normalization",
          title="spearman corr = "+str(np.round(spearman_corr,2)))
    ax.tick_params(left=True, bottom=True)
    
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.0, title="fraction", ncols=1)
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