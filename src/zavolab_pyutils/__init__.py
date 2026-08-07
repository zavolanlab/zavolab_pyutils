"""
zavolab_pyutils: Genomic data analysis utilities

A collection of utilities for common genomic data analysis tasks including
library size normalization, advanced Bayesian inference (Sanity), relative isoform 
usage testing, visualization, and read count simulations.
"""

from .general.read_count_data_analysis import (
    apply_deseq2_normalization,
    get_MultiDimR2,
    model_mean_variance,
    get_deseq2_means_and_errors,
    test_differential_expression,
)

from .general.annotation import (
    parse_gtf_attributes_into_pd_dataframes,
    genbank_to_fasta_and_gtf,
)

from .general.visualization import (
    plot_size_factors,
    pca_plot,
    plot_gene_expression_with_ci,
    plot_mean_variance_diagnostics,
    plot_mean_vs_cv,
    plot_expr_vs_libsize_correlation,
    horizontal_boxplot_with_stats,
)

from .general.read_count_simulation import (
    simulate_isoform_poisson_lognormal_counts,
    simulate_isoform_negative_binomial_counts,
)

from .pySanity.sanity import (
    apply_sanity_normalization,
    apply_sanity_normalization_full_bayesian,
    test_differential_relative_usage,
    prepare_isoform_sanity_matrix,
    test_differential_expression as sanity_test_differential_expression,
)

from .pySanity.frac_sanity import (
    fit_frac_sanity_map,
    calculate_differential_recruitment,
)

from .pySanity.visualization import (
    plot_sanity_gene_expression_with_ci,
    plot_sanity_relative_usage_with_ci,
    plot_variance_vs_expression,
    plot_frac_sanity_recruitment_with_ci,
    plot_beta_goodness_of_fit,
)

try:
    from importlib.metadata import version
    __version__ = version("zavolab_pyutils")
except Exception:
    __version__ = "unknown"
    
__author__ = "Zavolan Lab"
__license__ = "MIT"

__all__ = [
    # general
    "apply_deseq2_normalization",
    "get_MultiDimR2",
    "model_mean_variance",
    "get_deseq2_means_and_errors",
    "test_differential_expression",
    "parse_gtf_attributes_into_pd_dataframes",
    "genbank_to_fasta_and_gtf",
    "plot_size_factors",
    "pca_plot",
    "plot_gene_expression_with_ci",
    "plot_mean_variance_diagnostics",
    "plot_mean_vs_cv",
    "plot_expr_vs_libsize_correlation",
    "horizontal_boxplot_with_stats",
    "simulate_isoform_poisson_lognormal_counts",
    "simulate_isoform_negative_binomial_counts",
    # pySanity
    "apply_sanity_normalization",
    "apply_sanity_normalization_full_bayesian",
    "test_differential_relative_usage",
    "prepare_isoform_sanity_matrix",
    "fit_frac_sanity_map",
    "calculate_differential_recruitment",
    "plot_sanity_gene_expression_with_ci",
    "plot_sanity_relative_usage_with_ci",
    "plot_variance_vs_expression",
    "plot_frac_sanity_recruitment_with_ci",
    "plot_beta_goodness_of_fit",
]