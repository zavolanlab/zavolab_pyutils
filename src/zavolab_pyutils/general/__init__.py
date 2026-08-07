"""
general: General-purpose genomic data analysis utilities.
"""

from .read_count_data_analysis import (
    apply_deseq2_normalization,
    get_MultiDimR2,
    model_mean_variance,
    get_deseq2_means_and_errors,
    test_differential_expression,
)

from .annotation import (
    parse_gtf_attributes_into_pd_dataframes,
    genbank_to_fasta_and_gtf,
)

from .visualization import (
    plot_size_factors,
    pca_plot,
    plot_gene_expression_with_ci,
    plot_mean_variance_diagnostics,
    plot_mean_vs_cv,
    plot_expr_vs_libsize_correlation,
    horizontal_boxplot_with_stats,
)

from .read_count_simulation import (
    simulate_isoform_poisson_lognormal_counts,
    simulate_isoform_negative_binomial_counts,
)
