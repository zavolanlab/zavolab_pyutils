import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from adjustText import adjust_text
from scipy import stats

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
    
    xlabel = "# reads, mln"
    ylabel = "size factor,\nDeseq2 normalization"
    if log_scale:
        xlabel = xlabel + ", log2"
        ylabel = ylabel + ", log2"
    
    ax.set(xlabel=xlabel,
           ylabel=ylabel,
          title="spearman corr = "+str(np.round(spearman_corr,2)))
    ax.tick_params(left=True, bottom=True)
    
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.0, title="fraction", ncols=1)
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