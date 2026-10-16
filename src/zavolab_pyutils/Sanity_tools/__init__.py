"""
Sanity_tools: Python interface to the C++ Sanity normalisation (Breda et al., Nature
Biotechnology, 2021; https://github.com/jmbreda/Sanity, version >= 2.0).

The core fit of Sanity runs in the C++ binary; this subpackage writes its input, runs it, reads
its output and computes the summaries used downstream (log2 CPM-like expression, per-condition
means and standard errors, empirical Bayes priors on v_g, fracSanity inputs). Downstream tests,
fracSanity models and plots are shared with pySanity and re-exported here.

Install the binary with `make install-sanity` (see README).
"""

from .runner import (
    SanityNotFoundError,
    SanityOutput,
    find_sanity_binary,
    get_sanity_version,
    read_sanity_output,
    run_sanity,
    variance_grid,
    write_sanity_input,
)

from .normalization import (
    CPM_SCALE_FACTOR,
    SanityResult,
    apply_sanity_normalization,
    summarize_v_posterior,
)

from .frac import (
    prepare_frac_sanity_inputs,
    run_frac_sanity,
)

# downstream analyses shared with pySanity: they only use Sanity's outputs
from ..pySanity.sanity import (
    prepare_isoform_sanity_matrix,
    test_differential_expression,
    test_differential_relative_usage,
)

from ..pySanity.frac_sanity import (
    calculate_differential_recruitment,
    fit_frac_sanity_map,
)

from ..pySanity.frac_sanity_v2 import (
    beta_prior_from_interval,
    calculate_differential_recruitment_v2,
    fit_frac_sanity_v2,
)

from ..pySanity.visualization import (
    plot_beta_goodness_of_fit,
    plot_frac_sanity_recruitment_with_ci,
    plot_sanity_gene_expression_with_ci,
    plot_sanity_relative_usage_with_ci,
    plot_variance_vs_expression,
)

__all__ = [
    # running Sanity
    "SanityNotFoundError",
    "SanityOutput",
    "find_sanity_binary",
    "get_sanity_version",
    "read_sanity_output",
    "run_sanity",
    "variance_grid",
    "write_sanity_input",
    # normalisation
    "CPM_SCALE_FACTOR",
    "SanityResult",
    "apply_sanity_normalization",
    "summarize_v_posterior",
    # fracSanity
    "prepare_frac_sanity_inputs",
    "run_frac_sanity",
    "calculate_differential_recruitment",
    "fit_frac_sanity_map",
    "beta_prior_from_interval",
    "calculate_differential_recruitment_v2",
    "fit_frac_sanity_v2",
    # downstream tests
    "prepare_isoform_sanity_matrix",
    "test_differential_expression",
    "test_differential_relative_usage",
    # plots
    "plot_beta_goodness_of_fit",
    "plot_frac_sanity_recruitment_with_ci",
    "plot_sanity_gene_expression_with_ci",
    "plot_sanity_relative_usage_with_ci",
    "plot_variance_vs_expression",
]
