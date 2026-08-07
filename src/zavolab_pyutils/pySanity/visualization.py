"""
Visualization functions specific to pySanity outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from scipy import stats


def plot_sanity_gene_expression_with_ci(
    sample_norm_df, means_df, errors_df, metadata_df, selected_genes, 
    savefig_path, genes_df:pd.DataFrame=None, 
    sample_col='sample', cond_col='condition', 
    condition_order=None,palette=None,
    CI_limit=0.95, adjust_multiple_comparisons=False,
    mean_dot_size:float=6.0,sample_dot_size:float=4.0,
    one_subplot_width:float=2.8, subplot_height:float=5.2,
    add_text_labels_for_samples:bool=False,
    text_label_column_for_samples:str=None,
    text_label_size_for_samples:int=8
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
    genes_df : pd.DataFrame, optional
        DataFrame containing gene annotations. If provided, gene ids in the plot will be replaced with "gene_name". Default is None.
    adjust_multiple_comparisons : bool, optional
        If True, applies a Bonferroni correction to the CI width based on the 
        number of pairwise condition comparisons. Default is False.
    mean_dot_size : float, optional
        Size of the mean points in the plot. Default is 6.
    sample_dot_size : float, optional
        Size of the sample points in the plot. Default is 4.
    one_subplot_width : float, optional
        Width of a single subplot. Default is 2.8.
    subplot_height : float, optional
        Height of the subplot. Default is 5.2.
    add_text_labels_for_samples : bool, optional
        If True, adds text labels for each sample point in the plot. Default is False.
    text_label_column_for_samples : str, optional
        The column name in `metadata_df` to use for text labels when `add_text_labels_for_samples` is True. Default is None.
    text_label_size_for_samples : int, optional
        Font size for the sample text labels. Default is 8.
    """
    
    input_data_df = sample_norm_df.copy()
    input_data_df.index.name = 'index'
    
    common_genes = [g for g in selected_genes if g in input_data_df.index]

    melted = input_data_df.loc[common_genes].reset_index().rename(columns={'index': 'gene_name'})
    melted = pd.melt(melted, id_vars=['gene_name'], var_name=sample_col, value_name='log2_expr')
    
    # Select needed columns from metadata, ensuring the label column is fetched if requested
    meta_cols = [sample_col, cond_col]
    if add_text_labels_for_samples and (text_label_column_for_samples is not None):
        if text_label_column_for_samples not in meta_cols:
            meta_cols.append(text_label_column_for_samples)
            
    melted = pd.merge(metadata_df[meta_cols], melted, how='right', on=sample_col)
    
    if condition_order is None:
        order = sorted(melted[cond_col].unique())
    else:
        order = condition_order
    n_conditions = len(order)
    
    # Calculate Alpha with optional Bonferroni correction
    alpha_val = 1.0 - CI_limit
    if adjust_multiple_comparisons and n_conditions > 2:
        num_comparisons = (n_conditions * (n_conditions - 1)) / 2
        alpha_val /= num_comparisons
        
    z_score = stats.norm.ppf(1 - alpha_val / 2)
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(common_genes), sharey=True, figsize=(one_subplot_width*len(common_genes), subplot_height))
    if len(common_genes) == 1: axes = [axes]
        
    for k, gene in enumerate(common_genes):
        ax = axes[k]
        gene_data = melted[melted['gene_name'] == gene]

        if genes_df is not None and 'gene_id' in genes_df.columns and 'gene_name' in genes_df.columns:
            gene_name_to_plot = genes_df.loc[genes_df['gene_id'] == gene, 'gene_name'].values[0]
        else:
            gene_name_to_plot = gene

        y_pos = np.arange(len(order))
        log2_means = means_df.loc[gene, order].values
        err_margins = errors_df.loc[gene, order].values * z_score
        
        mean_data_df = pd.DataFrame([log2_means,order]).transpose()
        mean_data_df.columns = ['log2_expr',cond_col]

        ax = sns.pointplot(ax=ax, data=mean_data_df, x='log2_expr', y=cond_col, 
                             markersize=mean_dot_size,palette=palette, order=order,
                             color=('black' if palette is None else None), zorder=3, alpha=0.7,
                                errorbar=None,
                             )
        ax.errorbar(log2_means, y_pos, xerr=err_margins, fmt='o', color='black', capsize=4, zorder=2, markersize=0)
        ax = sns.swarmplot(
            ax=ax, data=gene_data, x='log2_expr', y=cond_col, order=order, 
            color='grey', size=sample_dot_size, edgecolor='black', linewidth=1, alpha=0.5, zorder=1,
        )
        
        # ADDED SECTION: Add text labels conditionally, iterating through drawn data
        if add_text_labels_for_samples and text_label_column_for_samples is not None:
            texts = []
            for index, row in gene_data.iterrows():
                # Skip instances where mapping isn't possible (missing data)
                if pd.isna(row['log2_expr']) or pd.isna(row[cond_col]):
                    continue
                    
                try:
                    # Y-coordinate relies on the condition's position in the ordered list
                    y_coord = order.index(row[cond_col])
                except ValueError:
                    continue
                    
                label_text = row[text_label_column_for_samples]
                if pd.isna(label_text):
                    continue
                    
                texts.append(
                    ax.annotate(
                        xy=(row['log2_expr'], y_coord),
                        text=str(label_text),
                        size=text_label_size_for_samples,
                        ha="left",
                    )
                )
            if texts:
                # Resolve overlapping elements cleanly leveraging adjustText library
                adjust_text(
                    texts, 
                    arrowprops=dict(arrowstyle='-', color='black', lw=0.5), 
                    ax=ax, 
                    expand_text=(1.2, 1.2),    # Multiplier for the text bounding box to ensure padding
                    expand_points=(1.2, 1.2),  # Multiplier for the point bounding box
                    force_text=(0.5, 0.5),     # Repulsion force between overlapping texts (x, y)
                    force_points=(0.5, 0.5),   # Repulsion force between text and data points (x, y)
                    max_iterations=2000        # Allows the algorithm more attempts to find a clean layout
                )
        
        ax.set(title=gene_name_to_plot, ylabel='', xlabel='$log_2~expr$')
        ax.tick_params(left=True, bottom=True)
        if k > 0: ax.tick_params(left=False)
        
    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_variance_vs_expression(means_df, vg_df, savefig_path, true_vg=None, ylim=None):
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
    ylim : tuple, optional
        Limits for the y-axis. Default is None.
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
    if ylim is not None:
        ax.set(ylim=ylim)
    fig.tight_layout()
    dir_path = Path(savefig_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_sanity_relative_usage_with_ci(
    norm_counts_df, variances_df, metadata_df, isoform_pairs_df, 
    savefig_path, sample_col='sample', cond_col='condition', 
    condition_order=None, palette=None,
    CI_limit=0.95, adjust_multiple_comparisons=False,
    log2_scale=True,
    mean_dot_size:float=6.0, sample_dot_size:float=4.0,
    one_subplot_width:float=2.8, subplot_height:float=5.2
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
    isoform_pairs_df : pd.DataFrame
        DataFrame containing 'isoform_numer' and 'isoform_denom' columns defining the numerator and denominator.
    savefig_path : str or pathlib.Path
        Output file path.
    sample_col : str, optional
        Column in metadata_df with sample IDs. Default is 'sample'.
    cond_col : str, optional
        Column in metadata_df with condition labels. Default is 'condition'.
    condition_order : list, optional
        Explicit list defining the order of conditions on the y-axis.
    palette : str or dict, optional
        Seaborn palette name or dictionary mapping conditions to colors.
    CI_limit : float, optional
        Confidence interval limit (e.g., 0.95 for 95% CI). Default is 0.95.
    adjust_multiple_comparisons : bool, optional
        If True, applies a Bonferroni correction to the CI width based on the 
        number of pairwise condition comparisons. Default is False.
    log2_scale : bool, optional
        If True (default), plots values on the log2 scale. 
        If False, values and error bars are exponentiated to the natural ratio scale.
    mean_dot_size : float, optional
        Size of the mean points in the plot. Default is 6.0.
    sample_dot_size : float, optional
        Size of the sample points in the plot. Default is 4.0.
    one_subplot_width : float, optional
        Width of a single subplot. Default is 2.8.
    subplot_height : float, optional
        Height of the subplot. Default is 5.2.
    """
    if 'isoform_numer' not in isoform_pairs_df.columns or 'isoform_denom' not in isoform_pairs_df.columns:
        raise ValueError("isoform_pairs_df must contain 'isoform_numer' and 'isoform_denom' columns.")

    sample_map = metadata_df.set_index(sample_col)[cond_col]
    
    if condition_order is None:
        order = sorted(sample_map.dropna().unique())
    else:
        order = condition_order
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
    for iso1, iso2 in zip(isoform_pairs_df['isoform_numer'], isoform_pairs_df['isoform_denom']):
        if iso1 in norm_counts_df.index and iso2 in norm_counts_df.index:
            valid_pairs.append((iso1, iso2))
        else:
            print(f"Warning: One or both isoforms not found in count matrix for pair ({iso1}, {iso2}). Skipping this pair.")
            
    if not valid_pairs:
        print("No valid isoform pairs found in the provided data.")
        return
        
    # Set up matplotlib figure dimensions
    fig, axes = plt.subplots(1, len(valid_pairs), sharey=True, figsize=(one_subplot_width * len(valid_pairs), subplot_height))
    if len(valid_pairs) == 1: 
        axes = [axes]
        
    for k, (iso1, iso2) in enumerate(valid_pairs):
        ax = axes[k]
        pair_name = f"{iso1}\nvs\n{iso2}"
        
        # 1. Calculate cell-level log2 ratios and aggregate posterior variances
        log2_ratio_cells = norm_counts_df.loc[iso1] - norm_counts_df.loc[iso2]
        var_ratio_cells = variances_df.loc[iso1] + variances_df.loc[iso2]
        
        # Prepare data for seaborn
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
        
        # --- Adjust for scale ---
        if log2_scale:
            plot_means = log2_means
            xerr = err_margins
            x_col = 'log2_ratio'
            x_label = '$log_2$(rel. usage ratio)'
        else:
            # Exponentiate the raw cell data for the swarmplot
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
            
        # Construct DataFrame for the pointplot
        mean_data_df = pd.DataFrame({x_col: plot_means, cond_col: order})
            
        # 3. Plotting
        # Draw the lines connecting condition means with pointplot
        ax = sns.pointplot(
            ax=ax, data=mean_data_df, x=x_col, y=cond_col, 
            markersize=mean_dot_size, palette=palette, order=order,
            color=('black' if palette is None else None), zorder=3, alpha=0.7,
            errorbar=None
        )
        
        # Draw the error bars representing the Bayesian Confidence Intervals
        # Note: markersize=0 so we don't draw over the pointplot marker
        ax.errorbar(plot_means, y_pos, xerr=xerr, fmt='o', color='black', capsize=4, zorder=2, markersize=0)
        
        # Overlay the individual cell points using swarmplot
        ax = sns.swarmplot(
            ax=ax, data=pair_data, x=x_col, y=cond_col, order=order, 
            color='grey', size=sample_dot_size, edgecolor='black', linewidth=1, alpha=0.5, zorder=1
        )
        
        # Define title if isoform names are present in the dataframe
        if 'isoform_name' in isoform_pairs_df.columns:
            pair_name = isoform_pairs_df.loc[
                (isoform_pairs_df['isoform_numer'] == iso1) & (isoform_pairs_df['isoform_denom'] == iso2)
            ]['isoform_name'].values[0]

        # Formatting
        ax.set(title=pair_name, ylabel='', xlabel=x_label)
        ax.tick_params(left=True, bottom=True)
        if k > 0: 
            ax.tick_params(left=False)
        
    # Save the figure
    Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)

def plot_frac_sanity_recruitment_with_ci(
    results_df: pd.DataFrame, 
    selected_genes: list, 
    savefig_path: str,
    CI_limit=0.95,
    mean_dot_size: float = 6.0,
    one_subplot_width: float = 2.8, 
    subplot_height: float = 5.2
):
    """
    Plots fracSanity log2(alpha) (recruitment fraction) with Bayesian CI error bars.
    """
    common_genes = [g for g in selected_genes if g in results_df['gene_id'].values]
    if not common_genes:
        print("No valid genes found.")
        return

    alpha_val = 1.0 - CI_limit
    z_score = stats.norm.ppf(1 - alpha_val / 2)
    
    sns.set(font_scale=1, style="white")
    fig, axes = plt.subplots(1, len(common_genes), sharey=True, 
                             figsize=(one_subplot_width * len(common_genes), subplot_height))
    if len(common_genes) == 1: 
        axes = [axes]
        
    conditions = ['UT', 'Stress']
    y_pos = np.arange(len(conditions))
    
    for k, gene in enumerate(common_genes):
        ax = axes[k]
        gene_data = results_df[results_df['gene_id'] == gene].iloc[0]
        
        log2_means = np.array([gene_data['log2_alpha_UT'], gene_data['log2_alpha_Stress']])
        
        # The variance of log2_alpha is V_g (since F is treated as a constant here)
        # Note: You will need to pass var_D_ut and var_D_stress in the dataframe for this
        err_margins = np.array([np.sqrt(gene_data['var_D_UT']), 
                                np.sqrt(gene_data['var_D_Stress'])]) * z_score
        
        # Plot lines
        ax.plot(log2_means, y_pos, color='grey', zorder=1, alpha=0.7)
        # Plot error bars
        ax.errorbar(log2_means, y_pos, xerr=err_margins, fmt='o', color='black', 
                    capsize=4, zorder=2, markersize=mean_dot_size)
        
        ax.set(title=gene, ylabel='', xlabel=r'$\log_2(\alpha)$ (Recruitment)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(conditions)
        
        ax.tick_params(left=True, bottom=True)
        if k > 0: 
            ax.tick_params(left=False)
            
    fig.tight_layout()
    fig.savefig(savefig_path, bbox_inches='tight', dpi=600)
    plt.close(fig)

def plot_beta_goodness_of_fit(alpha_vals: np.ndarray, a: float, b: float, condition_name: str, savefig_path: str):
    """
    Generates an eCDF vs theoretical CDF plot and a Q-Q plot to validate the Beta distribution assumption.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Remove out-of-bounds values for clean plotting
    clean_alpha = alpha_vals[(alpha_vals > 0) & (alpha_vals < 1)]
    
    # 1. ECDF vs CDF
    sorted_alpha = np.sort(clean_alpha)
    ecdf = np.arange(1, len(sorted_alpha) + 1) / len(sorted_alpha)
    theoretical_cdf = stats.beta.cdf(sorted_alpha, a, b)
    
    ax1.step(sorted_alpha, ecdf, label='Empirical CDF', color='black')
    ax1.plot(sorted_alpha, theoretical_cdf, label=f'Theoretical Beta({a:.2f}, {b:.2f})', color='red', linestyle='--')
    ax1.set_title(f'{condition_name}: eCDF vs Theoretical CDF')
    ax1.set_xlabel('Recruitment Efficiency (alpha)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.legend()
    
    # 2. Q-Q Plot
    theoretical_quantiles = stats.beta.ppf(ecdf, a, b)
    ax2.scatter(theoretical_quantiles, sorted_alpha, s=5, color='black', alpha=0.5)
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax2.set_title(f'{condition_name}: Q-Q Plot')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Empirical Quantiles')
    
    plt.tight_layout()
    plt.savefig(savefig_path, dpi=300)
    plt.close()
