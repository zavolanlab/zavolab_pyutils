"""
zavolab_pyutils: Genomic data analysis utilities

A collection of utilities for common genomic data analysis tasks including
library size normalization, advanced Bayesian inference (Sanity), relative isoform 
usage testing, visualization, and read count simulations.
"""

from .read_count_data_analysis import (
    apply_deseq2_normalization,
    apply_sanity_normalization_full_bayesian,
    prepare_isoform_sanity_matrix,
    test_differential_relative_usage,
    test_differential_expression
)

from .annotation import (
    parse_gtf_attributes_into_pd_dataframes, 
    genbank_to_fasta_and_gtf, 
)

from .visualization import (
    plot_size_factors,
    plot_sanity_gene_expression_with_ci,
    plot_sanity_relative_usage_with_ci
)

from .read_count_simulation import (
    simulate_isoform_poisson_lognormal_counts,
    simulate_isoform_negative_binomial_counts
)

try:
    from importlib.metadata import version
    __version__ = version("zavolab_pyutils")
except Exception:
    __version__ = "unknown"
    
__author__ = "Zavolan Lab"
__license__ = "MIT"

__all__ = [
    "apply_deseq2_normalization",
    "apply_sanity_normalization_full_bayesian",
    "prepare_isoform_sanity_matrix",
    "test_differential_relative_usage",
    "test_differential_expression",
    "convert_gff_to_gtf",
    "convert_gtf_to_gff",
    "parse_gtf_attributes",
    "plot_size_factors",
    "plot_sanity_gene_expression_with_ci",
    "plot_sanity_relative_usage_with_ci",
    "simulate_isoform_poisson_lognormal_counts",
    "simulate_isoform_negative_binomial_counts"
]