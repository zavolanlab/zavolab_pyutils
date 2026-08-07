"""
pySanity: Python implementation of the Sanity Bayesian normalization algorithm.
Original algorithm: Breda et al., Nature Biotechnology, 2021 (PMID: 33927416)
"""

from .sanity import (
    apply_sanity_normalization,
    apply_sanity_normalization_full_bayesian,
    test_differential_relative_usage,
    prepare_isoform_sanity_matrix,
    test_differential_expression,
)

from .frac_sanity import (
    fit_frac_sanity_map,
    calculate_differential_recruitment,
)

from .visualization import (
    plot_sanity_gene_expression_with_ci,
    plot_sanity_relative_usage_with_ci,
    plot_variance_vs_expression,
    plot_frac_sanity_recruitment_with_ci,
    plot_beta_goodness_of_fit,
)
